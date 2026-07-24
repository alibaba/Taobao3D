---
title: "TaoAvatar: Photorealistic 3D Avatars for Taobao Vision"
date: "2026-07-08"
description: "A year in review for TaoAvatar — from 3DGS static modeling and drivable avatars to 4DGS volumetric video, and how it powers Taobao Vision."
lang: "en"
slug: "taoavatar-summary"
---

TaoAvatar, our photorealistic 3D avatar technology, has been out for more than a year. This is a good moment to look back at the technical progress and the typical business scenarios we explored over the past year.

In terms of brand collaborations, the Taobao Vision Future Flagship Store's 3D smart shopping guide is now running steadily in Shanghai, Hangzhou, Nanjing, Changsha and other cities; the PYX (伯希和) virtual apparel store offers a 3D AI shopping guide and multimodal Q&A; and the team built apparel avatars for the Milan Winter Olympics, opening a virtual apparel store in Milan, Italy, with 3D clothing models on display.

<figure>
  <video src="/Taobao3D/blog/taoavatar/1.mp4" controls preload="metadata" playsinline></video>
  <figcaption>Taobao Vision Future Flagship Store — 3D smart shopping guide</figcaption>
</figure>

<figure>
  <video src="/Taobao3D/blog/taoavatar/2.mp4" controls preload="metadata" playsinline></video>
  <figcaption>Taobao Vision × PYX — 3D virtual apparel guide</figcaption>
</figure>

<figure>
  <video src="/Taobao3D/blog/taoavatar/3.mp4" controls preload="metadata" playsinline></video>
  <figcaption>Taobao Vision Milan Winter Olympics — 3D apparel avatars</figcaption>
</figure>

## 1. The many styles and use cases of digital humans

Digital humans come in many styles. There are the lightweight 2D avatars that are common today, as well as video avatars driven by video generation or video reenactment — recent work like Seedance and LPM has been very popular, showing rapid progress in the expressiveness and generation efficiency of video avatars. Looking further toward 3D, Meta's Codec Avatar is also moving toward large models for 3D generation and driving, pursuing lower-cost generation and deployment. Digital humans span anime, hyper-realistic, and photorealistic styles, with broad applications across film, gaming, communication, and e-commerce.

## 2. Overview of the TaoAvatar technology stack

In 2025, Taotian Group introduced TaoAvatar, a form of photorealistic 3D digital human, and delivered a complete experience on XR glasses. This year, TaoAvatar's focus has been twofold: first, continuing to expand the range of appearances and clothing it can support; and second, improving the richness and quality of drivable motions for a set of key avatar IPs.

Concretely, the team now has three main capabilities:

- **TaoModel — 3DGS static human modeling.** Through calibration, segmentation, geometry reconstruction, and Gaussian static reconstruction, it quickly produces high-quality static 3D human models. It helps business teams preview modeling results, and in certain scenarios can serve directly as the final deliverable.
- **TaoAvatar — 3DGS drivable avatar reconstruction.** Based on multi-view capture, SMPLX++ geometry reconstruction, and 3DGS dynamic reconstruction, it produces photorealistic 3D avatar assets drivable by voice, gesture, and motion. It still has some requirements on source material — relatively tight-fitting clothing, limited motion range, and clearly visible hands.
- **TaoVideo — 4DGS volumetric human video.** It supports capturing models with arbitrary clothing, hairstyles, and body types, and reconstructs motion along a timeline, faithfully capturing complex dynamics such as flowing skirts — opening up more creative space. This approach requires balancing reconstruction quality against storage cost.

![Overview of the TaoAvatar technology stack](../../../assets/blog/taoavatar/image1.webp)

## 3. Core technology

### 3.1 The photorealistic 3D avatar framework

TaoAvatar is a photorealistic 3D avatar solution built on 3D Gaussian Splatting. It addresses several typical problems of traditional 3D modeling: heavy computation, insufficient detail fidelity, and difficulty running on mobile devices.

![The photorealistic 3D avatar framework](../../../assets/blog/taoavatar/image2.webp)

In terms of capabilities, TaoAvatar integrates 3D Gaussian reconstruction, voice-driven lip sync, body-pose and gesture driving, on-device real-time rendering, and the MNN-LLM large language model inference engine. With these, it can serve e-commerce shopping guides, virtual companionship, and other scenarios with more lifelike avatars.

A quick note on why we chose XR devices for presentation: the human eye is naturally binocular, so XR devices better convey the stereoscopic sense of a 3D avatar. Among current headsets, the Apple Vision Pro performs well on resolution and stability — it has its own limitations, but for now it remains the most suitable device for presenting this kind of experience.

TaoAvatar's core metrics include: sharpness with PSNR above 35 and high stability; real-time, natural facial and body driving at 90 FPS; high-precision real-time rendering at 2K per eye and 90 FPS; and multimodal interaction with sub-2-second first-token latency and under 2.5 GB memory footprint. Overall the solution achieves 2K resolution, real-time driving and rendering at 90 FPS, production cost under ¥20,000, and delivery in under one week — and it has expanded from ToB reconstruction of specific IP characters to ToC construction of users' own digital doubles.

### 3.2 Body-shape and motion capture

TaoAvatar's first algorithmic core is body-shape and motion capture. It takes multi-view video as input and outputs SMPLX motion-capture data consistent with the subject's body shape and pose — data that can directly drive the 3DGS avatar.

![Body-shape and motion capture pipeline](../../../assets/blog/taoavatar/image3.webp)

The team built several extensions on top of SMPLX: SMPLX++ represents the non-rigid deformation of hair and clothing meshes; SMPLX-Ace performs automatic skeletal rigging; and retargeting transfers a mocap actor's motion onto the target avatar.

In the algorithm pipeline, layered supervision signals are extracted per frame: EMOCA for the face, HaMeR for the hands, and SAM-3DBody 3D keypoints for the body. The SMPLX initialization is optimized per frame, then mesh tracking combines image and supervision signals to ensure temporal smoothness and reduce jitter (the tracking model also provides rich semantic parsing). Finally, non-rigid clothing-mesh deformation is learned from images to produce the final SMPLX++ mocap data.

There are two key innovations here. First, decoupled layered supervision of face, hands, and body — each constrained separately before temporal tracking — makes the mocap results more stable. Second, small-studio / large-studio coordination: the large studio handles large-range motion driving, while the small studio adds non-rigid deformation, enabling consistent transfer and driving across studios.

For the studio coordination, the small studio first reconstructs a single frame to lock a unified skeleton and body shape, and the large studio then reuses that skeleton and shape to capture a wider range of motion. To improve tracking stability under sparse views, at least 5 full-body-visible views are required to stabilize body rigging and at least 3 hand-visible views to stabilize gesture rigging. On results, the average PVE error is 6–7 mm, outperforming the open-source SOTA framework EasyMocap with more complete mocap and more accurate body-shape estimation.

### 3.3 Dynamic human reconstruction

Next, TaoAvatar's approach to drivable 3DGS avatar reconstruction. The team's earlier method used a teacher-student framework, lightweight design, non-rigid deformation baking, and lightweight blendshape compensation — keeping avatar assets both sharp and compact enough for on-device real-time rendering.

The new challenge: how to support a single avatar ID being driven uniformly across different outfits and motions. To solve this, the team upgraded from "single person, single outfit, single motion clip" training to "single ID, multiple outfits, multiple motions" unified modeling.

![Single-ID, multi-outfit, multi-motion unified modeling](../../../assets/blog/taoavatar/image4.webp)

Concretely, multiple outfits and motion sequences are captured for the same ID from multi-view studio data, then SMPLX++ rigging and tracking are performed. Human-part parsing then finely segments body, clothing, and hair regions, and muscle and clothing non-rigid deformations are learned uniformly in UV space. Weights can be shared across outfits for cross-outfit reuse.

Finally, these deformations are baked via PCA into geometry and Gaussian deformation fields that run in real time on-device. At inference, feeding the ID's pose-driving signals produces natural motion across different outfits, including muscle and clothing deformation. The overall asset is around 400 MB and runs real-time inference and rendering at 90 FPS on-device.

The team showed the non-rigid deformation of clothing geometry across different poses, and the results of driving the same ID in different outfits with a single set of driving signals. Some clipping remains, but the driving results are largely consistent.

![Non-rigid deformation driving across different outfits](../../../assets/blog/taoavatar/image5.webp)

It also showed driving results for the same ID and outfit under different motions, covering scenarios such as walking guidance, interactive games, and in-store explanation.

![Driving results under different motions](../../../assets/blog/taoavatar/image6.webp)

### 3.4 Voice-driven gestures

Now TaoAvatar's voice- and text-driven gesture framework. In short, after receiving voice and text input, the system generates two types of motion: base gestures (natural rhythmic body responses while speaking) and strong-semantic gestures (specific actions triggered by the meaning of the text) — finally driving the 3DGS avatar via SMPLX.

![Voice- and text-driven gesture framework](../../../assets/blog/taoavatar/image7.webp)

Algorithmically, base gestures are generated mainly by GestureDiT, which combines voice and text features and models the SMPLX body hierarchically: it first generates upper-body motion, then predicts finger poses, with the lower body conditioned on the upper body. This makes motion in "talking in place" scenarios more natural and stable.

For strong-semantic gestures, the team uses Qwen-LLM to understand the text, retrieves matching pre-built semantic actions, and injects them into GestureDiT's denoising process, so the avatar produces better-matched gestures at key semantic moments.

For training, the team used monocular-video mocap data and 3D studio multimodal data. At inference, the model predicts 1 second of motion at a time and runs in real time (RTF < 1), deployable even on consumer GPUs like the 3090. The system generates matching gestures for the same semantics in both Chinese and English.

### 3.5 TaoVideo: photorealistic 4DGS volumetric video

Finally, TaoVideo-4DGS volumetric human video reconstruction. It takes multi-view video as input, places no restrictions on the subject or clothing, does not rely on explicit human models like SMPLX, and directly outputs a 4DGS dynamic reconstruction sequence.

![TaoVideo 4DGS volumetric video reconstruction](../../../assets/blog/taoavatar/image8.webp)

The core idea is to represent both the motion and appearance of Gaussians as spline curves. For motion, splines describe each Gaussian's position and rotation over time; for appearance, splines describe color changes, with a lifecycle mechanism indicating when a Gaussian appears and disappears. This better reproduces complex dynamics such as flowing clothes and swaying skirts.

The team also introduced non-uniform temporal control points and adaptive segmented modeling: regions with fast motion and rich detail get more control points and Gaussian capacity, while slower regions reduce redundancy — balancing quality and storage.

For training, a 20–25 second multi-view video is first initialized with static 3DGS, then dynamic detail is refined coarse-to-fine. At inference, on-device rendering reaches around 90 FPS, and a single asset is about 400 MB in FP16 storage. Overall, this approach suits video reconstruction of avatars with arbitrary clothing, complex appearance, and rich dynamics.

Subjectively, it stably renders the appearance detail and swaying deformation of loose garments such as hats, complex hairstyles, and skirts, with no obvious flickering inside the asset. On objective metrics, TaoVideo's average reconstruction PSNR is 31–35 and LPIPS 0.06–0.1. The team also showed a comparison between TaoVideo and TaoAvatar, especially in the dynamic behavior of skirts.

![TaoVideo vs. TaoAvatar comparison](../../../assets/blog/taoavatar/image9.webp)

## 4. Taobao Vision in the real world

Over the past year, the team extended the XR-glasses-based 3D Taobao experience from online to offline, opening Taobao Vision Future Flagship Stores in Hangzhou, Shanghai, Changsha, Shenzhen and more.

Taobao Vision (online) is a spatial shopping app built on Vision Pro, and won a 2025 Apple Design Award — the first time a Chinese internet platform has won in the award's 36-year history. Taobao Vision presents product shape, size, and material in immersive 3D, supports side-by-side comparison of multiple products to speed up purchase decisions, and — powered by in-house AI 3D reconstruction — offers natural interactions like voice and image search for a more intuitive, fluid shopping experience.

The Taobao Vision Future Flagship Store (offline) creates a new future shopping space that fuses "AI + 3D + XR" into an integrated solution for improving offline store efficiency: immersive display of massive SKUs in limited space, free mix-and-match for shoppers, and a more real, more engaging in-store experience. It is now open in Hangzhou, Shanghai, Changsha, Zhengzhou, Nanjing, and Shenzhen.

![Taobao Vision Future Flagship Store](../../../assets/blog/taoavatar/image10.webp)

## 5. What's next

Compared with studio-captured avatars, the team is also exploring low-cost avatar generation from a single image or sparse views, to further lower the barrier and scale up.

At CVPR this year the team proposed FHAvatar: rapidly reconstructing composable 3D Gaussian human heads from any number of views, modeling face and hair separately in texture space — planar Gaussians for the face and strand Gaussians for the hair. With just a few photos, it produces high-quality results in minutes, and supports real-time expression and lip driving, easy hairstyle swaps, and stylized editing.
