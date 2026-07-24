import { defineCollection, z } from 'astro:content';
import { file, glob } from 'astro/loaders';

// Publications — grouped in the UI by `category`.
const publications = defineCollection({
  loader: file('src/data/publications.yaml'),
  schema: z.object({
    id: z.string(),
    title: z.string(),
    venue: z.string(),
    category: z.enum(['recon', 'avatar']),
    highlight: z.boolean().default(false),
    links: z
      .array(
        z.object({
          label: z.string(),
          url: z.string().url().optional(), // no url => rendered as a plain "coming soon" tag
        })
      )
      .default([]),
  }),
});

// News timeline — sorted by date descending in the UI.
const news = defineCollection({
  loader: file('src/data/news.yaml'),
  schema: z.object({
    id: z.string(),
    date: z.string(), // ISO date, e.g. 2026-02-21
    text_en: z.string(),
    text_zh: z.string(),
  }),
});

// Open positions.
const jobs = defineCollection({
  loader: file('src/data/jobs.yaml'),
  schema: z.object({
    id: z.string(),
    name_zh: z.string(),
    name_en: z.string(),
    type: z.enum(['Intern', 'Full-time']),
    url: z.string().url(),
  }),
});

// Blog posts — one Markdown file per post, organised by locale under
// src/content/blog/<lang>/<slug>.md. `slug` pairs a post with its translation
// and controls the URL (/blog/<slug>, /en/blog/<slug>).
const blog = defineCollection({
  loader: glob({
    pattern: '**/*.md',
    base: 'src/content/blog',
    // Keep the locale folder in the id so zh/welcome and en/welcome don't collide.
    generateId: ({ entry }) => entry.replace(/\.md$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    date: z.string(), // ISO date, e.g. 2026-03-15
    description: z.string().default(''),
    lang: z.enum(['zh', 'en']),
    slug: z.string(),
  }),
});

export const collections = { publications, news, jobs, blog };
