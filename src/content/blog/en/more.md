---
title: "MoRE: When 3D Visual Geometry Reconstruction Meets Mixture-of-Experts (MoE)"
date: "2026-07-13"
description: "A walkthrough of the CVPR 2026 paper MoRE — bringing Mixture-of-Experts (MoE) to 3D visual geometry reconstruction to build a unified foundation model that jointly outputs point maps, depth, camera pose, tracking features and normals, refreshing several SOTA results with confidence-based depth refinement and dense semantic feature fusion."
lang: "en"
slug: "more"
---

> Paper: MoRE: 3D Visual Geometry Reconstruction Meets Mixture-of-Experts  
> Authors: Jingnan Gao, Zhe Wang, Xianze Fang, Xingyu Ren, Zhuo Chen, Shengqi Liu, Yuhao Cheng, Jiangjing Lyu, Xiaokang Yang, Yichao Yan  
> Affiliations: Shanghai Jiao Tong University, Alibaba Group  
> Venue: CVPR 2026

<figure>
  <video src="/Taobao3D/blog/more/video.mp4" controls preload="metadata" playsinline></video>
  <figcaption>MoRE demo</figcaption>
</figure>

## From "Per-Scene Optimization" to "Foundation Models"

Over the past few years, foundation models such as GPT, CLIP, DINO, and Stable Diffusion have proven one thing: as long as the data and model capacity are large enough, neural networks can learn general representations that transfer across tasks and scenes. 3D visual geometry reconstruction is undergoing the same paradigm shift — from "optimizing each scene individually" toward "feed-forward unified reconstruction." Models like DUSt3R, MASt3R, Fast3R, and VGGT have already shown that large-scale pretraining can endow a model with strong geometric priors.

But scaling 3D models further is not easy: geometric supervision signals are complex, real training data is noisy, and scene distributions are extremely diverse (indoor, outdoor, objects, people, dynamic). **MoRE**, jointly proposed by Shanghai Jiao Tong University and Alibaba, is designed precisely to break through these bottlenecks. It brings the **Mixture-of-Experts (MoE)** into 3D visual geometry reconstruction, building a unified foundation model that can simultaneously output point maps, depth maps, camera pose, tracking features, and surface normals.

## The Core Architecture of MoRE: Transformer + MoE

MoRE uses a dense visual Transformer as its backbone and extends a series of geometry prediction heads with an MoE mechanism. The input is a set of RGB images with unknown camera parameters, and the outputs include:

- Camera intrinsics and extrinsics (Camera)
- Per-pixel 3D point map (Pointmap)
- Depth map (Depth)
- Tracking features (Tracking)
- Surface normals (Normal)

![MoRE overall architecture](../../../assets/blog/more/architecture.webp)
*Fig 1. MoRE adopts two-stage training: the first stage trains a dense Transformer with multi-task objectives; the second stage introduces the Mixture-of-Experts (MoE), letting different experts specialize in different scenes and tasks for scalable and efficient geometry prediction.*

The core idea of the MoE layer is "conditional computation": for each token, a router predicts which experts it should be sent to, then only the top-K experts are activated for a weighted aggregation. In this way, the model's total parameter count can be large, while the parameters activated in each forward pass remain few. More importantly, different experts can naturally divide the work — some excel at indoor structures, some at large outdoor scenes, some specialize in object details — improving adaptability to diverse 3D data. To keep expert load balanced, MoRE also introduces a differentiable load-balancing loss, avoiding a situation where all tokens flood toward a few "popular" experts.

For training stability, MoRE is initialized from VGGT pretrained weights and uses an adaptive loss-truncation strategy: it maintains the mean and standard deviation of recent losses, and when a step's loss exceeds the μ + 3σ threshold it is clipped, preventing occasional outlier samples from dominating the gradient. The training data covers indoor, outdoor, object-/person-centric, and dynamic scenes, ensuring the model's generalization in real applications.

## Two Key Techniques: Making Real Data and Fine Geometry More Reliable

### 1. Confidence-based Depth Refinement

Real-world depth training data often contains noise or missing values. Supervising directly with such labels makes the model prone to overfitting to wrong depths. MoRE's approach is: first predict depth with the current strongest monocular depth model, MoGev2, then compare it against the ground-truth depth to generate a confidence mask M_conf, keeping only high-confidence regions for supervision.

![Confidence-based depth refinement comparison](../../../assets/blog/more/depth_refinement.webp)
*Fig 2. With the confidence mask, the model can ignore noise in the ground-truth depth labels (such as the red regions), yielding cleaner and more accurate depth estimates.*

This prior-guided depth loss, combined with the original multi-view depth loss, significantly improves the stability and accuracy of depth estimation.

### 2. Dense Semantic Feature Fusion

Multi-view reconstruction models often tend to produce smooth, consistent but detail-poor geometry; monocular models are rich in detail but lack global consistency. MoRE fuses the globally aligned 3D backbone features with each image's DINOv2 dense semantic features, then feeds them into the normal prediction head. This way, the model preserves multi-view consistency while capturing fine surface structures.

![Semantic feature fusion ablation](../../../assets/blog/more/ablation_normal.webp)
*Fig 3. After adding dense semantic features (w/ f_s), normal prediction is noticeably clearer and more accurate in detailed regions such as fur, metal, and fabric.*

## Experimental Results: Refreshing SOTA on Multiple Benchmarks

MoRE was systematically evaluated on tasks including point map estimation, monocular depth estimation, camera pose estimation, and surface normal estimation, achieving leading or highly competitive results on all of them.

![Qualitative comparison of multi-view reconstruction](../../../assets/blog/more/qualitative_comparison.webp)
*Fig 4. Qualitative comparison with methods such as Pi3, VGGT, and Fast3R shows that MoRE reconstructs more complete and consistent geometry across scenes like the Great Wall, meeting rooms, sofas, and race cars.*


![Depth refinement ablation](../../../assets/blog/more/ablation_depth.webp)
*Fig 5. The ablation study further shows that the confidence-based depth refinement module keeps depth estimation more stable in specular, transparent, or low-texture regions.*

## Final Thoughts

MoRE demonstrates the great potential of the MoE architecture in 3D visual geometry reconstruction: it can not only "trade fewer activated parameters for larger model capacity" like large language models, but also let different experts automatically adapt to diverse 3D scenes. Combined with confidence-based depth refinement and dense semantic feature fusion, MoRE refreshes state-of-the-art results on multiple mainstream benchmarks, providing a more scalable and robust 3D visual foundation model for applications such as AR/VR, game content generation, robotic perception, and autonomous driving.
