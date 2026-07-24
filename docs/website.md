# 官网技术实现文档

淘天 Meta 技术团队官网 —— 基于 Astro 构建的静态站点，部署于 GitHub Pages。

---

## 1. 概述

- **定位**：团队门户单页站，内容来源于仓库根目录 `README.md`（结构化后迁入站点数据）。
- **形态**：单页长滚动 + 顶部锚点导航，中英双语，中文为默认语言。
- **访问地址**：`https://alibaba.github.io/Taobao3D/`（项目站，带 `/Taobao3D` 子路径）。
- **不含**：`Dens3R` / `HRM2Avatar` 分支的代码——官网仅通过外链引用其 Project Page。

---

## 2. 技术选型

| 层 | 技术 | 版本 | 说明 |
|---|---|---|---|
| 框架 | [Astro](https://astro.build) | `^5.12` | 静态优先，构建产物纯静态 HTML/CSS，零默认 JS |
| 样式 | [Tailwind CSS](https://tailwindcss.com) | `^4.1` | 通过 `@tailwindcss/vite` 插件接入（v4 无需 `tailwind.config.js`）|
| 内容 | Astro Content Collections | 内置 | 用 `file()` loader 加载 YAML，`zod` 校验 schema |
| 国际化 | Astro i18n 路由 | 内置 | `astro:i18n` + 自建文案字典 |
| 部署 | GitHub Actions + Pages | `withastro/action@v3` | 推送 `main` 自动构建发布 |

选型理由：内容体量小、以展示为主，静态站最简单可靠；Astro 的 Content Collections + i18n 让"加论文/改文案/切语言"都只改数据文件。

---

## 3. 目录结构

```
Taobao3D/
├─ astro.config.mjs          # 站点配置：base 路径、i18n、Tailwind
├─ package.json              # 依赖与脚本
├─ tsconfig.json             # 继承 astro/tsconfigs/strict
├─ .github/workflows/deploy.yml   # CI 部署
├─ public/                   # 原样拷贝的静态资源（带 base 前缀访问）
│  ├─ overview.png           # Hero 配图（已压缩 5.7MB→1.1MB）
│  ├─ favicon.svg
│  └─ jobs/*.png             # 招聘海报（当前未在页面引用，备用）
└─ src/
   ├─ content.config.ts      # Content Collections schema 定义
   ├─ data/                  # 结构化内容数据
   │  ├─ publications.yaml    #   论文列表
   │  ├─ news.yaml            #   动态（双语）
   │  └─ jobs.yaml            #   招聘岗位（双语）
   ├─ i18n/ui.ts             # 界面文案字典（zh/en）+ 翻译工具
   ├─ styles/global.css      # Tailwind 引入 + 品牌主题变量
   ├─ layouts/Base.astro     # 页面骨架：<head>/SEO/Header/Footer
   ├─ components/            # 各区块组件
   │  ├─ Header.astro         #   固定导航 + 语言切换
   │  ├─ Hero.astro           #   首屏
   │  ├─ About.astro          #   愿景 / 研究方向
   │  ├─ Demo.astro           #   效果展示（iframe 嵌入）
   │  ├─ News.astro           #   动态时间线
   │  ├─ Publications.astro    #   论文卡片
   │  ├─ Jobs.astro           #   招聘表格
   │  └─ Footer.astro         #   页脚
   └─ pages/
      ├─ index.astro          # 中文首页（默认，根路径）
      └─ en/index.astro       # 英文首页（/en/）
```

---

## 4. 核心配置（astro.config.mjs）

```js
export default defineConfig({
  site: 'https://alibaba.github.io',
  base: '/Taobao3D',                 // 项目站子路径
  trailingSlash: 'ignore',
  i18n: {
    defaultLocale: 'zh',             // 中文默认
    locales: ['zh', 'en'],
    routing: { prefixDefaultLocale: false },  // 中文不带 /zh/ 前缀
  },
  vite: { plugins: [tailwind()] },
});
```

**关键点：`base` 子路径**
GitHub 项目站运行在 `/Taobao3D/` 下，因此所有站内绝对路径必须带该前缀，否则线上 404。约定：

- 站内资源（图片、favicon）：用 `import.meta.env.BASE_URL` 拼接，例：
  ```js
  const overview = `${base}/overview.png`.replace(/\/+/g, '/');
  ```
  `replace(/\/+/g,'/')` 折叠多余斜杠，兼容 `base='/Taobao3D'` 与将来自定义域名 `base='/'` 两种情况。
- 站内路由（语言切换）：用 `getRelativeLocaleUrl()`（见 §6），自动带 base。
- 站外链接（arXiv、Project Page、GitHub）：直接写完整 URL。

> 若将来启用自定义域名：把 `base` 改回 `'/'`、`site` 改为该域名，并在 `public/` 加 `CNAME` 文件。其余代码无需改动。

---

## 5. 内容管理（Content Collections）

内容与展示分离：数据在 `src/data/*.yaml`，schema 在 `src/content.config.ts`。

### Schema 定义（src/content.config.ts）

```ts
const publications = defineCollection({
  loader: file('src/data/publications.yaml'),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    venue: z.string(),
    category: z.enum(['recon', 'avatar']),   // 决定分组
    highlight: z.boolean().default(false),   // true=金色 badge（如 CVPR Highlight）
    links: z.array(z.object({
      label: z.string(),
      url: z.string().url().optional(),      // 无 url → 渲染为"即将公开"占位
    })).default([]),
  }),
});
// news:  { id, date, text_en, text_zh }
// jobs:  { id, name_zh, name_en, type: Intern|Full-time, url }
```

`file()` loader 要求 YAML 中每条记录含唯一 `id` 字段。

### 论文标题的语言策略

论文标题按学术惯例**保持英文**，不做翻译；仅分组小标题（"三维重建与生成" 等）随语言切换。News、Jobs 则提供 `_zh`/`_en` 双字段。

---

## 6. 国际化（i18n）

### 路由

- `prefixDefaultLocale: false` → 中文页在根 `/Taobao3D/`，英文页在 `/Taobao3D/en/`。
- 对应文件：`src/pages/index.astro`（zh）与 `src/pages/en/index.astro`（en），两者仅 `lang` 常量与 import 深度不同，复用同一批组件。

### 文案字典（src/i18n/ui.ts）

所有界面文案集中于此，按 `zh` / `en` 两套结构化对象组织：

```ts
export const ui = {
  zh: { teamName, nav:{...}, hero:{...}, about:{...}, demo:{...}, ... },
  en: { ... },
} as const;

export function useTranslations(lang: Lang) {
  return ui[lang] ?? ui[defaultLang];
}
```

组件通过 props 接收 `lang`，取出对应文案：

```astro
---
import { useTranslations, type Lang } from '../i18n/ui';
const { lang } = Astro.props;
const t = useTranslations(lang);
---
<h2>{t.about.heading}</h2>
```

### 语言切换按钮（Header.astro）

用 `astro:i18n` 的 `getRelativeLocaleUrl` 生成带 base 的目标地址，当前语言高亮：

```ts
import { getRelativeLocaleUrl } from 'astro:i18n';
const switchLinks = [
  { code: 'zh', label: '中文', href: getRelativeLocaleUrl('zh') },  // → /Taobao3D/
  { code: 'en', label: 'EN',   href: getRelativeLocaleUrl('en') },  // → /Taobao3D/en/
];
```

切换跳转到对方语言的首页（不保留当前锚点位置）。

---

## 7. 组件与页面

页面区块顺序（`index.astro`）：

```
Hero → About(愿景) → News(动态) → Demo(效果) → Publications(论文) → Jobs(加入)
```

导航项：**愿景 · 动态 · 效果 · 论文 · 加入**（英文 Vision · News · Demos · Publications · Join），锚点分别对应 `#about #news #demo #publications #jobs`。

### 效果展示区块（Demo.astro）

以 `<iframe>` 嵌入外部 3D demo 页面：

```astro
const demoUrl = 'https://market.m.taobao.com/app/T3dShared/acennr-engine-examples/index.html';

<iframe src={demoUrl} title={t.demo.heading} loading="lazy"
        referrerpolicy="no-referrer"
        allow="xr-spatial-tracking; accelerometer; gyroscope; camera; fullscreen"
        class="h-full w-full border-0" />
```

- 窗口为**与页面同宽**（`max-w-6xl`）的圆角悬浮卡片，高度 `80vh`（限 520–820px），带品牌色柔光。
- `loading="lazy"`：滚动到视口才加载，避免拖慢首屏。
- **可嵌性前提**（已实测该 URL 通过）：目标页无 `X-Frame-Options` / CSP `frame-ancestors` 限制、无 JS 反嵌入、HTTPS。
- ⚠️ `market.wapa.taobao.com` 的 **wapa 为淘宝预发环境**，长期使用建议替换为正式生产 URL——只需改 `demoUrl` 一处。

---

## 8. 样式（Tailwind v4）

`src/styles/global.css`：

```css
@import "tailwindcss";

@theme {
  --font-sans: "Inter", ..., "PingFang SC", "Microsoft YaHei", sans-serif;
  --color-brand-50:  #eef4ff;
  --color-brand-100: #d9e6ff;
  --color-brand-500: #3b6ff6;   /* 主品牌蓝 */
  --color-brand-600: #2b57d4;
  --color-brand-700: #1f42a8;
}

html { scroll-behavior: smooth; scroll-padding-top: 5rem; }  /* 锚点跳转避开固定头 */
body { @apply bg-white text-slate-800 antialiased; }
```

- 品牌色通过 `@theme` 定义为 `--color-brand-*`，组件里用 `bg-brand-500`、`text-brand-700`、`from-brand-50`、`bg-brand-500/10`（透明度修饰符自动合成）等。
- **已定义的品牌色阶**：50 / 100 / 500 / 600 / 700。使用其它阶（如 brand-200/400）需先在 `@theme` 补充。

---

## 9. 静态资源

- 放 `public/` 的文件按原路径拷贝到构建产物，访问时需带 base 前缀（见 §4）。
- `overview.png` 原始 5.7MB（4019px 宽），已用 `sips` 压至 1600px 宽 / 1.1MB。
- `public/jobs/*.png`：招聘海报，当前页面用表格呈现、未引用这些图，作备用保留。

---

## 10. 本地开发与构建

```bash
npm install          # 安装依赖
npm run dev          # 开发服务器 → http://localhost:4321/Taobao3D/
npm run build        # 生产构建 → dist/
npm run preview      # 预览构建产物（同样在 /Taobao3D/ 子路径下）
```

构建产物 `dist/` 含 `index.html`（中文）与 `en/index.html`（英文），已通过 base 子路径验证。

---

## 11. 部署（GitHub Pages）

`.github/workflows/deploy.yml`：推送到 `main` 触发。

```
build 作业:  checkout → withastro/action@v3（自动装依赖/build/上传 artifact）
deploy 作业: actions/deploy-pages@v4 → 发布到 Pages
```

**首次上线前置操作**（需仓库管理员）：
1. 仓库 Settings → Pages → Source 选 **GitHub Actions**。
2. 提交并推送本站点文件到 `main`（当前均为未跟踪状态，未推送则不会触发部署）。

发布地址：`https://alibaba.github.io/Taobao3D/`。

---

## 12. 常见维护操作

| 需求 | 操作 |
|---|---|
| 加一篇论文 | 在 `src/data/publications.yaml` 增一条（含唯一 `id`、`category`、`links`）|
| 加一条动态 | 在 `src/data/news.yaml` 增一条（`date` + `text_zh`/`text_en`），按日期自动倒序 |
| 加/改招聘岗位 | 编辑 `src/data/jobs.yaml`（`name_zh`/`name_en`/`type`/`url`）|
| 改界面文案 | 编辑 `src/i18n/ui.ts` 对应 `zh`/`en` 字段 |
| 换 Demo 地址 | 改 `src/components/Demo.astro` 的 `demoUrl` 常量 |
| 换 Hero 配图 | 替换 `public/overview.png`（建议压缩后 <1.5MB）|
| 调整区块顺序 | 改 `src/pages/index.astro` 与 `src/pages/en/index.astro`（两处需同步）|
| 新增一种语言 | astro.config `locales` 加项 → `ui.ts` 补该语言字典 → 加 `src/pages/<lang>/index.astro` |
| 改为自定义域名 | astro.config `base='/'` + `site` 改域名 + `public/CNAME` |

---

## 13. 已知注意事项

1. **base 子路径**是最易踩的坑：本地根路径正常但线上 `/Taobao3D/` 下 404，均因硬编码了 `/`——站内资源务必用 `BASE_URL` 拼接。
2. **Demo 用的是预发（wapa）URL**，稳定性不保证，建议尽快替换为正式 URL。
3. **中英首页需同步**：`index.astro` 与 `en/index.astro` 增删区块要一起改（内容差异不会被测试发现）。
4. **页脚年份**当前硬编码为 2026；若需自动更新可改为 `new Date().getFullYear()`（静态构建期取值）。
5. 中文导航项名（如"愿景"）与其区块标题（"研究方向"）不完全一致，属有意的短导航设计。
