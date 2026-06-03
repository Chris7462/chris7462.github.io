// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Yi-Chen Zhang',
  tagline: 'Perception & Robotics Engineer — Autonomous Systems',
  favicon: 'img/yc.ico',

  future: {
    v4: true,
  },

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

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: './sidebars.js',
        },
        blog: {
          showReadingTime: true,
          blogSidebarTitle: 'All posts',
          blogSidebarCount: 'ALL',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
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
