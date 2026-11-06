---
title: "TaoFlowForge: Progressive Native Mesh Generation via Cascaded Flow Matching"
date: "2026-09-30"
description: "A walkthrough of the TaoFlowForge technical report — a foundation model for production-ready 3D mesh generation that decomposes the pipeline into three cascaded stages: coarse vertex generation, late-stage progressive refinement, and joint connectivity-normal prediction, achieving state-of-the-art results among open-source native mesh generators."
lang: "en"
slug: "taoflowforge"
---

> Paper: TaoFlowForge: Progressive Native Mesh Generation via Cascaded Flow Matching  
> Authors: Xianze Fang\*, Qiyuan Feng\*, Dongfang Sun, Yan Zhang, Xiuchao Wu, Jingnan Gao, Jiangjing Lyu†, Chengfei Lyu†, Gang Yu  
> Affiliation: Taobao3D Team, Alibaba Group

## 1. Introduction: Why Is Production-Ready Mesh Generation Hard?

Triangle meshes are the fundamental basis of modern 3D production pipelines — rigging, animation, rendering engines, and real-time deployment all depend on them. Image-conditioned 3D mesh generation models can greatly simplify the workflows for artists and designers, but not all generated meshes are ready for industrial deployment.

Models based on SDF (Signed Distance Function) representations typically produce meshes with excessively high face counts and topologically irregular structures, posing three major challenges:

- **UV unwrapping difficulty**: complex topologies bottleneck the entire production pipeline.
- **Storage overhead**: heavy assets hinder real-time on-device rendering.
- **Irregular topology**: makes downstream rigging and animation extremely complicated.

These issues are especially acute on large-scale e-commerce platforms like Taobao, where 3D assets serve as the interactive content for immersive VR shopping (such as the Vision Pro release of Taobao) and on-device product display and AR placement on mobile devices.

Recent work has explored two alternative approaches — autoregressive methods that generate vertices and faces sequentially, and diffusion-based methods that generate 3D primitives directly. However, autoregressive methods are computationally inefficient and discard intrinsic 3D structural information, while existing diffusion-based methods still suffer from unstable vertex generation or the lack of explicit face orientation prediction.

**TaoFlowForge** is designed to address all these challenges. It is a foundation model for native production-ready 3D content generation that produces lightweight, editable, and topologically clean meshes.

![TaoFlowForge generates production-ready native meshes across categories (furniture, characters, props), rendered with their predicted topology. The framed images show the UV unwrappings, revealing the quality of their topology.](../../../assets/blog/taoflowforge/teaser.webp)

## 2. Method Overview: A Three-Stage Cascaded Pipeline

TaoFlowForge decomposes the mesh generation process into **vertex generation** and **connectivity prediction**, organized as three cascaded image-conditioned stages:

![Overview of TaoFlowForge: Stage 1 generates coarse-quantized vertices at 64³ resolution; Stage 2 progressively refines vertex positions from 64³ to 512³; Stage 3 predicts vertex connectivity to form mesh faces with correct orientations.](../../../assets/blog/taoflowforge/pipeline.webp)

### Stage 1: Coarse Structure Generation

Given an input image, TaoFlowForge first generates coarse-quantized vertex positions at 64³ resolution. An occupancy VAE compresses the 64³ voxel grid into a compact latent space, and a 1.3B-parameter latent flow DiT generates the coarse structure conditioned on frozen DINOv3 image features. This stage is initialized from pretrained weights (Trellis.2), effectively leveraging learned priors for predicting coarse object structure from images.

### Stage 2: Late-Stage Progressive Refinement

This is where TaoFlowForge truly differentiates itself. Stage 2 operates directly in the uncompressed occupancy space (no VAE), employing three parameter-independent transformer experts for the 64³→128³, 128³→256³, and 256³→512³ transitions. Each expert specializes in its own resolution level.

The key innovation is **normalized raw-space flow matching**: since the binary occupancy target's statistics shift sharply with resolution, TaoFlowForge whitens each level with its own Bernoulli statistics, ensuring the target has zero mean and unit variance — matching the Gaussian source distribution.

### Stage 3: Joint Connectivity and Normal Prediction

Given the refined vertices from Stage 2, Stage 3 predicts both vertex connectivity (edges) and per-vertex normals simultaneously. A topology VAE learns a continuous feature for each vertex, and a latent flow DiT generates these features conditioned on vertex coordinates via voxel RoPE.

The connectivity head uses an elegant **asymmetric bilinear scoring** design: instead of a single symmetric embedding (which would incorrectly encourage transitive connectivity), each vertex is projected into two distinct subspaces — "source" and "destination" — before computing a symmetrized score. This prevents hallucinated spurious edges.

Face orientation is determined by comparing the geometric normal of each recovered triangle against the predicted vertex normals, avoiding the need for post-processing correction.

## 3. Key Technical Innovations

### 3.1 Topology-Aware Losses for Vertex Generation

Standard per-cell binary cross-entropy treats each voxel independently, ignoring mesh connectivity. TaoFlowForge introduces two geometry-aware co-occupancy priors:

![Topology-aware co-occurrence priors: edge and face priors activate only when all connected vertices are jointly predicted empty, reinforcing structurally important high-degree vertices.](../../../assets/blog/taoflowforge/loss_diagrams.webp)

- **Edge endpoint prior**: a soft-OR penalty that grows only when both endpoints of a true edge are predicted empty — injecting mesh connectivity as an occupancy prior. Importantly, this naturally assigns larger weights to high-degree vertices (vertices with more neighbors), enhancing supervision of structurally critical points.
- **Face corner prior**: extends the same soft-OR logic to the three corners of each ground-truth face.

Combined with boundary-aware weighted binary cross-entropy (which up-weights the thin surface band where false positives concentrate), these losses ensure that the generated vertex set faithfully preserves the mesh's structural integrity.

### 3.2 Late-Stage Refinement vs. Direct Generation

A key design choice is the two-stage coarse-to-fine vertex generation. The alternative — directly generating 512³ vertex support in a single stage — degrades both global structure and fine detail. The progressive approach preserves the global layout established by Stage 1 while Stage 2 adds fine vertices within that support, yielding more stable structures than direct high-resolution generation.

### 3.3 Large-Scale Data Curation

TaoFlowForge is trained on a large and diverse mesh dataset (~800K meshes) from three sources:

1. **Taobao in-house 3D dataset**: everyday indoor commodities and character models.
2. **Manually crafted in-house dataset**: fills sparse categories.
3. **Public datasets**: Objaverse and TexVerse.

A multi-stage curation pipeline filters the raw data: face-count bucketing for balanced sampling, topology statistics screening (non-manifold faces, vertex degree distribution, spatial uniformity), and finally a VLM-based agent that inspects rendered views to prune residual low-quality cases.

## 4. Experimental Results

### 4.1 Quantitative Comparison

TaoFlowForge is evaluated on the public Toys4K benchmark and TE-388 (a dataset of 388 manually crafted examples spanning indoor, outdoor, and character categories) against two representative open-source native mesh generators: LATO.2 (diffusion-based) and EdgeRunner (autoregressive).

Under single-image conditioning, TaoFlowForge achieves **state-of-the-art results on all six metrics** — Chamfer Distance (CD), Hausdorff Distance (HD), ULIP-I, Uni3D-I, FD-Inception, and FD-DINOv2 — on both benchmarks. For example, on Toys4K, TaoFlowForge reduces CD from 0.1553 (LATO.2) to **0.0655** and FD-DINOv2 from 616.42 to **197.73**.

### 4.2 Qualitative Comparison with Open-Source Methods

![Qualitative comparison with open-source polygon-based 3D mesh generation models. Given an image as input, TaoFlowForge generates the finest and most complete 3D meshes.](../../../assets/blog/taoflowforge/main_compare.webp)

EdgeRunner (autoregressive) tends to output smoothed convex hulls with irregular triangulation and often drops thin or topologically isolated structures — limbs go missing, hair collapses into a single shell, and intricate facial features are washed out. LATO.2 (diffusion) recovers global shape well, but TaoFlowForge produces cleaner local triangulation and better preserves thin, branchy parts.

### 4.3 Comparison with Commercial Models

![Qualitative comparison with commercial polygon-based 3D mesh generation models. TaoFlowForge generates more detailed and stable meshes with better alignment.](../../../assets/blog/taoflowforge/second_compare.webp)

TaoFlowForge remains highly competitive with leading proprietary commercial models in terms of overall geometry quality, fine-structure preservation, and topological fidelity.

### 4.4 More Results

![Additional image-conditioned results. TaoFlowForge generates meshes with both strong global structural stability and high-fidelity local details.](../../../assets/blog/taoflowforge/gallery.webp)

### 4.5 Ablation Studies

![Qualitative ablation of late-stage progressive refinement. Replacing the late-stage refinement strategy with direct continuous upsampling degrades both the overall geometry and the fidelity of fine details.](../../../assets/blog/taoflowforge/ablation.webp)

The ablation studies validate two key design choices:

- **Late-stage progressive refinement**: replacing the coarse-to-fine cascade with direct single-stage generation degrades both global structure (e.g., skateboard geometry) and local details (e.g., wheel fidelity).
- **Normalized targets and geometry-aware losses**: removing either component leads to measurable drops across all six evaluation metrics.

## 5. Conclusion

TaoFlowForge presents a progressive native mesh generation framework developed as the foundation model of the Taobao 3D pipeline. Through a three-stage cascaded design — coarse structure generation, late-stage progressive refinement, and joint connectivity-normal prediction — it produces production-ready meshes with clean topology, correct face orientations, and fine geometric details.

The combination of topology-aware losses, asymmetric bilinear edge scoring, and a carefully curated 800K-mesh training dataset enables TaoFlowForge to achieve state-of-the-art performance among open-source methods and remain competitive with proprietary commercial methods. Looking ahead, TaoFlowForge can be applied to more general generation pipelines, such as serving as a component of large-scale multimodal models, generating 3D scenes based on scene graphs, or producing UV-space textures.
