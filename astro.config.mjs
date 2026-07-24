// @ts-check
import { defineConfig } from 'astro/config';
import tailwind from '@tailwindcss/vite';

// GitHub Pages project site:  https://alibaba.github.io/Taobao3D/
// `base` MUST match the repo name so assets resolve under the sub-path.
// If a custom domain is added later, set base to '/' and site to that domain.
export default defineConfig({
  site: 'https://alibaba.github.io',
  base: '/Taobao3D',
  trailingSlash: 'ignore',
  // Chinese is the default locale, served at the root (/Taobao3D/).
  // English lives under /Taobao3D/en/.
  i18n: {
    defaultLocale: 'zh',
    locales: ['zh', 'en'],
    routing: {
      prefixDefaultLocale: false,
    },
  },
  vite: {
    plugins: [tailwind()],
  },
});
