---
title: "MatMart: PBR Material Reconstruction of 3D Objects via Diffusion"
date: "2026-07-13"
description: "A walkthrough of the CVPR 2026 paper MatMart — a single diffusion model that performs two-stage PBR material reconstruction, using VMCA and progressive inference for high-fidelity, scalable results with O(1) memory complexity independent of input view count."
lang: "en"
slug: "matmart"
---

> Paper: MatMart: Material Reconstruction of 3D Objects via Diffusion  
> Authors: Xiuchao Wu, Pengfei Zhu, Jiangjing Lyu, Xinguo Liu, Jie Guo, Yanwen Guo, Weiwei Xu, Chengfei Lyu  
> Affiliations: Zhejiang University, Nanjing University, Alibaba Group  
> Venue: CVPR 2026

## 1. Introduction: Why Is Material Reconstruction Hard?

Recovering the material of a 3D object from RGB images (Material Reconstruction) is a long-standing challenge at the intersection of computer vision and graphics. Unlike plain texturing, PBR materials require decomposing surface reflectance into physical parameters such as **Albedo, Roughness, and Metallic**, and ensuring that these parameters render results consistent with real-world physics across different viewpoints and lighting conditions.

Traditional methods rely mainly on differentiable rendering for per-object optimization, which often requires capturing a large number of images and tends to be inefficient and unstable. Recently, diffusion models have shown strong potential for material estimation and generation, but existing approaches still face three core challenges:

- **High-fidelity detail preservation**: Semantic information such as text, logos, and fine textures on an object's surface is difficult for existing generative models to faithfully reproduce.
- **Scalability**: Real applications need a framework that can handle an arbitrary number of input images and support high-resolution inference, yet existing methods often cannot scale due to memory constraints.
- **System simplicity**: Systems that depend on multiple pretrained models or multi-stage cascades add training and deployment complexity and may introduce instability.

MatMart is a unified solution proposed precisely for these problems.

![Figure 1: MatMart's reconstruction results from single-view input (source: paper Figure 1)](../../../assets/blog/matmart/page-01.webp)

## 2. Overview: A Two-Stage Unified Diffusion Framework

The core idea of MatMart is to split material reconstruction into two complementary stages, both completed within a single diffusion model:

![Figure 2: Overview of MatMart. Stage 1 performs progressive material estimation and bakes it into UV space; Stage 2 generates and bakes unobserved regions guided by priors (source: paper Figure 2)](../../../assets/blog/matmart/page-03.webp)

### Stage 1: Progressive Material Estimation

Given a set of RGB images, the model progressively estimates the PBR material corresponding to each image and bakes the results into UV space. The goal of this stage is to recover the material details of the input images as accurately as possible in the **visible regions**.

### Stage 2: Prior-guided Material Generation

For single-view or sparse-view inputs, occluded or unobserved regions are inevitable in UV space. The second stage uses the material baked out in the first stage as a prior, adaptively selects new viewpoints, and alternates between generation and baking to progressively complete the missing regions.

The advantage of the two stages is **estimate first, then generate**. Stage 1 is responsible for faithfully preserving input details, while Stage 2 leverages diffusion priors to complete unknown regions — each with its own role, yet sharing the same model.


## 3. Key Techniques: VMCA and Progressive Inference

### 3.1 View-Material Cross-Attention (VMCA)

To keep multi-view consistency during progressive inference, the authors propose **View-Material Cross-Attention (VMCA)**.

Traditional cross-view attention needs to concatenate all input images at once, with a spatial complexity of O(N²), making it hard to handle high resolution or a large number of inputs. MatMart adopts **round-by-round progressive inference**, processing only one Target View and one Reference View per round. The reference view is **the prediction/generation result from the previous round**, so information is passed in a chain across rounds. VMCA builds a connection between the target view and the reference view:

- Query comes from the target view;
- Key and Value come from the concatenation of the target view and the reference view;
- The reference view itself is not contaminated by the target view and is output directly as Value.

Since the number of views participating in attention per round is fixed at 2, VMCA has a spatial complexity of **O(1)**, independent of the total number of input images. This means MatMart can support inputs ranging from a single image to arbitrarily many images while keeping memory usage stable.

![Figure 3: Effect of VMCA on the consistency of progressive material estimation. With VMCA, predictions of the same object across different views are more consistent (source: paper Figure 3)](../../../assets/blog/matmart/page-04.webp)

### 3.2 Adaptive View Selection and Alternating Baking

In the second stage, MatMart does not blindly generate all views but instead performs adaptive view selection based on the coverage of texels in UV space:

1. Use the 6 axis-aligned direction views as base viewpoints;
2. Uniformly sample 300 candidate viewpoints on the sphere;
3. Use a greedy strategy, selecting the viewpoint that maximizes UV coverage each time;
4. Stop when coverage reaches the threshold ρ=0.95 or the number of views reaches the upper limit N=10.

At each generation step, besides the material prior from the first stage, the network input also includes **geometric priors** rendered from the known geometry (Normal and Point Position), as well as a Generation Mask identifying the region to be generated. To improve efficiency, the authors use depth-based warping to quickly obtain these priors and masks.

The basic unit of VMCA is still "target view + reference view," but in Stage 2, for efficiency, **3 target views are grouped together** for parallel inference (group size can be adjusted based on memory). After each round of generation, the results are projected back to UV space and used to update the existing texture via a weighted blending strategy. The weights consider the cosine similarity between the view direction and the surface normal, reducing the contribution of regions observed at large angles and thus minimizing baking artifacts.

### 3.3 Unified Architecture: One Model for Both Prediction and Generation

Another highlight of MatMart is completing **prediction and generation end-to-end with a single diffusion model**. Both tasks output PBR materials and both use VMCA to ensure consistency, so they can share the same U-Net backbone.

Concretely, building on a pretrained Stable Diffusion, each attention block contains three layers of attention:

- **Cross-component Attention**: exchanges information between Albedo and Roughness/Metallic;
- **View-Material Cross-Attention (VMCA)**: ensures multi-view consistency;
- **Text-prompt Cross-Attention**: uses text prompts to control whether the current output is Albedo or RM.

For the different input forms of the two tasks, the authors use zero padding to align tensor shapes. The zero padding also serves as a task identifier, letting the model distinguish whether it is currently doing prediction or generation.

## 4. Training and Inference Details

### Training Setup

- Datasets: ABO, G-Objaverse, Arb-Objaverse (consistent with IDArb);
- Hardware: 16 NVIDIA H20 GPUs;
- Optimizer: AdamW, learning rate 1×10⁻⁴;
- Prediction target: v-prediction;
- Training strategy: prediction and generation tasks are optimized alternately; since the first view of progressive estimation has no reference view, the authors also add training for prediction without a reference view;
- Resolution: first trained at 256×256 for 20K steps, then at 512×512 for 50K steps.

### Inference Setup

- A single V100 GPU is enough for inference;
- Inference resolution: 1024×1024;
- Meshes without UVs are unwrapped using Blender Smart UV Project;
- Inference time per object is about 9–23 minutes, depending on the number of selected views N.

## 5. Experimental Results

### 5.1 Quantitative Comparison

The authors selected 100 objects from an Objaverse subset, rendering 9 views per object, and tested using 1 and 3 images as input respectively. Compared methods include NvDiffRec, Paint3D, Material Anything, MaterialMVP, and the Stage1+TexGEN combination.

Under both single-view and multi-view settings, MatMart (1024×1024) achieves leading metrics:

- **Albedo**: SSIM and PSNR are clearly better than the baselines;
- **Roughness/Metallic**: lower MSE;
- **Rendering realism**: FID and LPIPS drop significantly.

Notably, high-resolution inference (1024×1024) brings further improvements over 512×512, verifying that MatMart can still effectively align geometry and texture details at high resolution.

### 5.2 Qualitative Comparison

From the visualizations, MatMart better preserves the semantic details in the input images, such as the icon on a blue bag or the text on a vase. In comparison:

- NvDiffRec struggles to disentangle material from lighting under sparse views, and its renderings are noisy;
- Paint3D tends to bake shadows and reflections into textures;
- MaterialMVP has decent visual quality but loses input details;
- Material Anything shows seams or inconsistencies across multiple views;
- TexGEN generates directly in UV space and, limited by network resolution, is prone to blur and color errors.

![Figure 5: Qualitative comparison under single-view input. MatMart recovers more accurate materials on a variety of objects, with renderings closer to the Ground Truth (source: paper Figure 5)](../../../assets/blog/matmart/page-06.webp)

Tests on the real-world dataset Stanford-ORB also show that MatMart generalizes well.

![Figure 6: Multi-view material reconstruction results on the real-world dataset Stanford-ORB (source: paper)](../../../assets/blog/matmart/com_real_multi.webp)


### 5.3 Ablation Studies

The authors ablate three core designs:

- **VMCA**: removing it noticeably degrades multi-view consistency and worsens the final rendering quality;
- **Material Priors**: if the material prior provided by Stage 1 is blacked out, the generated results show clear inconsistencies across views;
- **Stage1 Baking**: skipping the first stage and generating from scratch drastically reduces reconstruction quality, showing that the estimate-first-then-generate strategy is essential.

In addition, the experiments find that as the number of input views increases, reconstruction quality keeps improving but saturates at around 10 views; meanwhile, GPU memory usage stays at about 24 GB regardless of the number of input views, fully demonstrating the framework's scalability.

![Figure 7: VMCA ablation — removing it noticeably degrades multi-view consistency (source: paper)](../../../assets/blog/matmart/vmca.webp)
![Figure 8: Material prior ablation — blacking out the prior causes clear inconsistencies across views (source: paper)](../../../assets/blog/matmart/ablation_mat.webp)
![Figure 9: Stage1 Baking ablation — generating directly without the first stage drastically reduces reconstruction quality (source: paper)](../../../assets/blog/matmart/nobaking.webp)
![Figure 10: Effect of the number of input views on reconstruction quality and memory usage (source: paper)](../../../assets/blog/matmart/numview.webp)

## 6. Limitations and Future Directions

The authors also candidly acknowledge the current limitations in their conclusion:

1. **Inherent ambiguity of material decomposition**: this may cause an overall color scaling in the predicted albedo;
2. **Objects with strong self-occlusion**: more views may be needed to achieve satisfactory generation results.

These issues are shared challenges across the entire inverse-rendering field and are worth further exploration in future research.

## 7. Conclusion

Through the combination of **two-stage reconstruction + progressive inference + VMCA + a unified single model**, MatMart offers a solution for PBR material reconstruction of 3D objects that balances high fidelity, scalability, and ease of deployment. It faithfully preserves input details in visible regions while leveraging diffusion priors to complete unknown regions, and achieves O(1) memory complexity independent of the number of input views. For researchers and engineers working on 3D reconstruction, inverse rendering, and digital asset generation, MatMart presents a new paradigm worth attention.
