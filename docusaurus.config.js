// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Yi-Chen Zhang',
  tagline: 'Perception & Robotics Engineer — Autonomous Systems',
  favicon: 'img/yc.ico',

  url: 'https://chris7462.github.io',
  baseUrl: '/',

  organizationName: 'Chris7462',
  projectName: 'chris7462.github.io',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          blogSidebarTitle: 'All posts',
          blogSidebarCount: 'ALL',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    [
      '@docusaurus/plugin-content-blog',
      {
        id: 'music',
        path: 'music',
        routeBasePath: 'music',
        blogTitle: 'Music',
        blogDescription: 'Piano pieces by Yi-Chen Zhang',
        blogSidebarTitle: 'All pieces',
        blogSidebarCount: 'ALL',
        showReadingTime: false,
        authorsMapPath: '../blog/authors.yml',
      },
    ],
  ],

  themeConfig:
    ({
      image: '/img/profile.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: '',
        logo: {
          alt: 'Yi-Chen Zhang Logo',
          src: 'img/yc.png',
        },
        items: [
          {
            href: 'https://github.com/Chris7462',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://www.linkedin.com/in/yi-chen-zhang-b72907116/',
            label: 'LinkedIn',
            position: 'right',
          },
          {
            href: 'pathname:///cv/cv.pdf',
            label: 'CV',
            position: 'right',
          },
          {
            to: '/docs',
            label: 'Docs',
            position: 'left',
          },
          {
            to: '/blog',
            position: 'left',
            label: 'Blog',
          },
          {
            to: '/music',
            position: 'left',
            label: 'Music',
          },
        ],
      },
      footer: {
        style: 'dark',
        copyright: `© ${new Date().getFullYear()} Yi-Chen Zhang. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
