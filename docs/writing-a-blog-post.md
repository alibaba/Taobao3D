# 如何新增一篇博客

本文档说明如何为 Taobao3D 官网新增一篇博客文章。博客采用 Astro Content Collections + Markdown，中英双语，中文为默认语言。

---

## 1. 快速上手（TL;DR）

新增一篇博客 = 在 `src/content/blog/` 下建**两个** Markdown 文件（中、英各一），共用同一个 `slug`：

```
src/content/blog/
├─ zh/my-first-post.md      # 中文版
└─ en/my-first-post.md      # 英文版
```

两个文件的 frontmatter 里 `slug` 必须相同（这样中英切换才能配对），`lang` 分别填 `zh` / `en`。保存后本地 `npm run dev` 即可看到，无需改动任何代码。

---

## 2. 目录与命名约定

- **位置**：`src/content/blog/<lang>/<文件名>.md`
  - `<lang>` 只能是 `zh` 或 `en`。
  - 文件名建议用英文短横线 kebab-case（如 `on-device-rendering.md`），与 `slug` 保持一致，便于维护。
- **URL 由 `slug` 决定**，与文件名无关，但建议二者一致：
  - 中文：`/Taobao3D/blog/<slug>/`
  - 英文：`/Taobao3D/en/blog/<slug>/`
- **中英配对靠 `slug`**：语言切换按钮会跳到另一语言下 `slug` 相同的文章。

> ⚠️ 一定要放进 `zh/` 或 `en/` 子目录。加载器用完整路径生成 id，若把 `welcome.md` 直接放在 `blog/` 根目录，会与其他语言的同名文件冲突。

---

## 3. Frontmatter 字段

每个 Markdown 文件顶部的 YAML frontmatter 字段如下（schema 定义在 `src/content.config.ts` 的 `blog` 集合）：

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `title` | ✅ | string | 文章标题，显示在列表卡片、文章页 `<h1>` 和浏览器 `<title>` |
| `date` | ✅ | string | ISO 日期，如 `2026-07-08`。列表按此**倒序**排列 |
| `description` | ❌ | string | 摘要。显示在列表卡片和文章页副标题；缺省为空 |
| `lang` | ✅ | `zh` \| `en` | 语言，必须与所在目录一致 |
| `slug` | ✅ | string | URL 片段 + 中英配对键。同一篇文章的中英版本必须相同 |

示例：

```markdown
---
title: "端侧 3D 引擎的内存管理实践"
date: "2026-07-15"
description: "在受限的端侧设备上，如何让 3D 引擎的内存占用可预测、可回收。"
lang: "zh"
slug: "on-device-memory"
---

## 引言

正文从这里开始……
```

---

## 4. 正文写法

正文是标准 Markdown，构建时渲染为 HTML，并套用站点的文章排版样式 `.blog-prose`（定义在 `src/styles/global.css`）。已适配的元素：

- 标题 `##` / `###`（正文里**不要**再用 `#`，`title` 已作为 `<h1>`）
- 段落、**加粗**、*斜体*、[链接](https://example.com)
- 有序 / 无序列表
- 引用块 `>`
- 行内 `code` 与代码块（带语法高亮容器）

代码块示例：

````markdown
```glsl
vec3 color = lighting(albedo, normal, lightDir);
fragColor = vec4(toneMap(color), 1.0);
```
````

图片：把图片放到 `public/` 下，用带 base 前缀的绝对路径引用，例如：

```markdown
![示意图](/Taobao3D/blog/my-post/diagram.png)
```

---

## 5. 完整操作步骤

1. **建中文文件** `src/content/blog/zh/<slug>.md`，填好 frontmatter（`lang: "zh"`）与正文。
2. **建英文文件** `src/content/blog/en/<slug>.md`，`slug` 与中文一致（`lang: "en"`）。
   - 如暂时只有中文，可先只建中文版；英文列表就不显示这篇，语言切换到英文会落回英文博客列表。
3. **本地预览**：
   ```bash
   npm run dev
   ```
   打开 `http://localhost:4321/Taobao3D/blog/` 查看列表，点进文章确认排版。
4. **构建校验**（可选，推送前建议跑一次）：
   ```bash
   npm run build
   ```
   若 frontmatter 字段缺失或类型不对，`zod` 会在构建时报错。
5. **提交推送**：
   ```bash
   git add src/content/blog
   git commit -m "blog: 新增《<标题>》"
   git push
   ```
   推送后 CI 自动构建并发布到 GitHub Pages。

---

## 6. 常见问题

- **文章不显示？** 检查文件是否在 `zh/` 或 `en/` 子目录、`lang` 是否与目录一致、`date` 是否为合法 ISO 日期。
- **中英切换跳错/跳回列表？** 两个语言版本的 `slug` 必须完全相同。
- **构建报 schema 错误？** 对照第 3 节确认必填字段齐全、`lang` 只填 `zh`/`en`。
- **想改列表卡片 / 文章页样式？** 列表在 `src/components/Blog.astro`，文章排版在 `src/styles/global.css` 的 `.blog-prose`，详情页模板在 `src/pages/blog/[slug].astro` 与 `src/pages/en/blog/[slug].astro`。

---

## 7. 相关文件一览

| 作用 | 路径 |
|---|---|
| 集合定义 / schema | `src/content.config.ts`（`blog` 集合） |
| 文章内容 | `src/content/blog/<lang>/<slug>.md` |
| 列表组件 | `src/components/Blog.astro` |
| 列表页 | `src/pages/blog.astro`、`src/pages/en/blog.astro` |
| 详情页（动态路由） | `src/pages/blog/[slug].astro`、`src/pages/en/blog/[slug].astro` |
| 文章排版样式 | `src/styles/global.css`（`.blog-prose`） |
| 导航与文案 | `src/components/Header.astro`、`src/i18n/ui.ts`（`nav.blog` / `blog.*`） |
