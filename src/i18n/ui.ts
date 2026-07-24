// Central UI dictionary. Chinese (zh) is the default locale; English (en) mirrors it.
// Structured content (publications / news / jobs) lives in src/data/*.yaml.

export const languages = { zh: '中文', en: 'English' } as const;
export const defaultLang = 'zh';
export type Lang = keyof typeof languages;

export const ui = {
  zh: {
    brand: 'Taobao3D',
    teamName: '淘天技术团队',
    teamSub: 'TaoTian Technology Team',
    nav: {
      home: '首页',
      demo: '演示',
      news: '动态',
      blog: '博客',
      publications: '论文',
      jobs: '加入',
      github: 'GitHub',
    },
    hero: {
      badge: '3D · XR',
      description:
        '一键实现高质量 3D 内容生成，开启 3D/XR 电商购物新体验。',
      ctaDemo: '体验 Demo',
      ctaPubs: '查看动态',
      ctaJoin: '加入我们',
    },
    about: {
      heading: '研究方向',
      intro: '我们的工作覆盖 3D 与 XR 全栈，从算法研发延展到引擎与应用。',
      areas: [
        {
          title: '3D 大模型',
          desc: '聚焦通用品类的高还原度 3D 模型生成。',
          img: 'aigc3d.webp',
          points: [
            '3D 几何重建：实现高还原度、细节丰富的雕塑级 3D Mesh 生成，几何细节一致性优异，还原效果接近真实物理结构；',
            '材质大模型：基于 3D 原生材质大模型的 PBR 材质解耦，输出高分辨率纹理贴图，可精准还原文字、Logo 等精细信息，同时支持透明与半透明材质的精确建模；',
            '多模态 3D 生成：融合文本、图片及视频等多模态输入，更精准地引导 3D 内容生成，显著提升模型的泛化能力与几何还原度（包括结构比例、遮挡区域补齐等）；',
            '网格生成：在原生三维坐标空间中直接生成专业级 3D 网格模型，具备组件化建模能力，满足工业级精度需求。',
          ],
        },
        {
          title: '3D 真人数字人',
          desc: '聚焦高保真、可驱动、易编辑的 3DGS 数字人生成与交互。',
          img: 'TaoAvatar.webp',
          points: [
            '3DGS 可驱动重建：融合多视角采集、SMPL-X 几何先验与 3DGS 动态表征，构建支持语音、手势、动作等多模态驱动的 3D 真人数字人资产；',
            '4DGS 体积视频：突破服饰和发型限制，通过时序动态重建完整捕捉裙摆、发丝、宽松服装等复杂物理运动，提升动态表现力与创意自由度；',
            '前馈式 3D 数字人：支持单图/多图前馈推理，实现骨骼动画自然驱动、面部换妆、身体换衣等一体化编辑能力；',
            '业务应用：以 3D 导购数字人形态服务淘宝 Vision 未来旗舰店，以 3D 服饰模特数字人形态合作伯希和、米兰冬奥 - 李宁等品牌。',
          ],
        },
        {
          title: '端侧 3D 引擎',
          desc: '自研跨平台、功能完整好用且久经业务验证的 3D 引擎。',
          img: 'acennr.webp',
          points: [
            '运行时：支持 Metal、Vulkan、OpenGL、WebGL 等主流图形后端，并提供 TypeScript 接口降低业务开发门槛；',
            '工作流：配套完整的生产工作流，从辅助建模插件到模型自动优化管线，与运行时紧密协同，实现高品质 3D 素材的规模化生产；',
            '3D Agent：借助 VLM、2D/3D 生成算法以及 AI Coding 技术，构建 Agentic 3D 工作流 加速低门槛、轻互动但高质量的 3D 应用开发；',
            '业务应用：已落地于库鸥（3D 模型预览）、淘宝人生（敦煌博物馆、迪士尼 100 周年、淘宝人生 2）、手猫 3D、天猫新品、Taobao Vision、超影计划等多项业务。',
          ],
        },
      ],
    },
    demo: {
      heading: '效果展示',
      intro: '在浏览器中实时体验我们的典型示例，包含 3D 模型、3D 场景和 3D 数字人。',
      disclaimer: '注：上述 3D 效果仅用于学术研究与技术展示，不涉及任何商业用途，仅供技术参考。',
    },
    news: { heading: '最新动态' },
    blog: {
      heading: '博客',
      intro: '技术分享、研究解读与团队思考。',
      readMore: '阅读全文 →',
      back: '← 返回博客',
      empty: '博客文章即将上线，敬请期待。',
    },
    publications: {
      heading: '论文成果',
      groupRecon: '三维重建与生成',
      groupAvatar: '三维人体 / 虚拟形象建模与动画',
      soon: '即将公开',
    },
    jobs: {
      heading: '淘天集团 - 技术团队',
      intro: '负责面向消费场景的 3D 基础技术建设和 XR 创新应用探索，打造以手机及 XR 新设备为载体的消费购物新体验。',
      colPosition: '职位',
      colType: '类型',
      colLink: '链接',
      apply: '投递 →',
      typeIntern: '实习',
      typeFulltime: '正式',
    },
    footer: { rights: '淘天技术团队' },
  },

  en: {
    brand: 'Taobao3D',
    teamName: 'TaoTian Technology Team',
    teamSub: '淘天技术团队',
    nav: {
      home: 'Home',
      demo: 'Demos',
      news: 'News',
      blog: 'Blog',
      publications: 'Publications',
      jobs: 'Join',
      github: 'GitHub',
    },
    hero: {
      badge: '3D · XR',
      description:
        'We are dedicated to advancing and applying 3D and XR technologies, delivering high-performance, scalable solutions that enable next-generation digital experiences.',
      ctaDemo: 'Try Demo',
      ctaPubs: 'View News',
      ctaJoin: 'Join Us',
    },
    about: {
      heading: 'What we work on',
      intro:
        'Our work spans the full 3D and XR stack, from algorithm research to engines and applications.',
      areas: [
        {
          title: '3D Foundation Model',
          desc: 'High-fidelity 3D model generation for general product categories.',
          img: 'aigc3d.webp',
          points: [
            'Geometry reconstruction: sculpture-grade 3D mesh generation with rich detail and excellent geometric consistency, closely matching real physical structure.',
            'Material foundation model: PBR material decomposition via a 3D-native material foundation model, producing high-resolution texture maps that faithfully reproduce fine details such as text and logos, with accurate modeling of transparent and translucent materials.',
            'Multimodal 3D generation: fusing text, image, and video inputs to guide 3D content generation more precisely, significantly improving generalization and geometric fidelity (including structural proportions and occluded-region completion).',
            'Polygonal mesh generation: directly generating professional-grade 3D meshes in native 3D coordinate space, with component-based modeling for industrial-grade precision.',
          ],
        },
        {
          title: 'Photorealistic 3D Avatars',
          desc: 'High-fidelity, drivable, and easily editable 3DGS avatar generation and interaction.',
          img: 'TaoAvatar.webp',
          points: [
            '3DGS drivable reconstruction: fusing multi-view capture, SMPL-X geometry priors, and 3DGS dynamic representation to build photorealistic 3D avatar assets driven by voice, gesture, and motion.',
            '4DGS volumetric video: going beyond clothing and hairstyle constraints, capturing complex physical motion of skirts, hair strands, and loose garments via temporal dynamic reconstruction for greater expressiveness and creative freedom.',
            'Feed-forward 3D avatars: single- and multi-image feed-forward inference, enabling integrated editing such as natural skeletal animation, face makeup, and clothing swaps.',
            'Business applications: serving the Taobao Vision Future Flagship Store as 3D shopping-guide avatars, and partnering with brands such as PYX and Milan Winter Olympics · Li-Ning as 3D apparel-model avatars.',
          ],
        },
        {
          title: 'On-device 3D Engine',
          desc: 'An in-house, cross-platform 3D engine — full-featured, easy to use, and battle-tested across many businesses.',
          img: 'acennr.webp',
          points: [
            'Runtime: supports major graphics backends including Metal, Vulkan, OpenGL, and WebGL, with a TypeScript API to lower the barrier for business development.',
            'Workflow: a complete production pipeline, from modeling-assistant plugins to automatic model optimization, working closely with the runtime to enable scalable production of high-quality 3D assets.',
            "3D Agent: powered by VLMs, AI coding, and 2D/3D generation, and built on AceNNR's in-house engine and workflow, the 3D Agent accelerates low-barrier, lightweight-interaction yet high-quality 3D application development.",
            'Business applications: shipped across many businesses, including Kuou (3D model preview), Taobao Life (Dunhuang Museum, Disney 100th Anniversary, Taobao Life 2), Mobile Tmall 3D, Tmall New Products, Taobao Vision, and the Chaoying Project.',
          ],
        },
      ],
    },
    demo: {
      heading: 'Demos',
      intro:
        'Experience our examples live in the browser — including models, scenes, and digital humans.',
      disclaimer:
        'Note: the 3D demos above are for academic research and technical demonstration only. They are not intended for any commercial use and are provided solely for technical reference.',
    },
    news: { heading: 'News' },
    blog: {
      heading: 'Blog',
      intro: 'Technical deep-dives, research notes, and team perspectives.',
      readMore: 'Read more →',
      back: '← Back to blog',
      empty: 'Blog posts are coming soon — stay tuned.',
    },
    publications: {
      heading: 'Publications',
      groupRecon: '3D Reconstruction and Generation',
      groupAvatar: '3D Human / Avatar Modeling and Animation',
      soon: 'soon',
    },
    jobs: {
      heading: 'TaoTian · Technology Team',
      intro:
        'Building foundational 3D technology for consumer scenarios and exploring innovative XR applications, creating new shopping experiences on phones and next-generation XR devices.',
      colPosition: 'Position',
      colType: 'Type',
      colLink: 'Link',
      apply: 'Apply →',
      typeIntern: 'Intern',
      typeFulltime: 'Full-time',
    },
    footer: { rights: 'TaoTian Technology Team · 淘天技术团队' },
  },
} as const;

export function useTranslations(lang: Lang) {
  return ui[lang] ?? ui[defaultLang];
}
