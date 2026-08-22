/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* eslint-disable */

const {themes} = require('prism-react-renderer');
const lightCodeTheme = themes.github;
const darkCodeTheme = themes.dracula;

const isPublicBuild = process.env.SUPERDEX_PUBLIC_BUILD === '1';
const publicOrigin = (
  process.env.SUPERDEX_PUBLIC_ORIGIN || 'https://projectsuperdex.com'
).replace(/\/+$/, '');
const publicBasePath = (
  process.env.SUPERDEX_PUBLIC_BASE_URL || '/'
).replace(/^\/+|\/+$/g, '');
const projectBaseUrl = publicBasePath ? `/${publicBasePath}/` : '/';
const siteUrl = isPublicBuild ? publicOrigin : 'https://internalfb.com';
const siteBaseUrl = isPublicBuild ? `${projectBaseUrl}studio/` : '/';
const projectSuperdexUrls = isPublicBuild
  ? {
      landing: `${publicOrigin}${projectBaseUrl}`,
      physics: `${publicOrigin}${projectBaseUrl}physics/`,
      robotics: `${publicOrigin}${projectBaseUrl}robotics/`,
      lab: `${publicOrigin}${projectBaseUrl}lab/`,
      studio: `${publicOrigin}${projectBaseUrl}studio/`,
    }
  : {
      // @oss-disable: landing: 'https://www.internalfb.com/intern/staticdocs/superdex/',
      landing: `${publicOrigin}${projectBaseUrl}`, // @oss-enable
      // @oss-disable: physics: 'https://www.internalfb.com/intern/staticdocs/mochi_physics/',
      physics: `${publicOrigin}${projectBaseUrl}physics/`, // @oss-enable
      // @oss-disable: robotics: 'https://www.internalfb.com/intern/staticdocs/superdex_robotics/',
      robotics: `${publicOrigin}${projectBaseUrl}robotics/`, // @oss-enable
      // @oss-disable: lab: 'https://www.internalfb.com/intern/staticdocs/superdex_lab/',
      lab: `${publicOrigin}${projectBaseUrl}lab/`, // @oss-enable
      // @oss-disable: studio: 'https://www.internalfb.com/intern/staticdocs/superdex_studio/',
      studio: `${publicOrigin}${projectBaseUrl}studio/`, // @oss-enable
    };
const projectSuperdexUrl = projectSuperdexUrls.landing;

// With JSDoc @type annotations, IDEs can provide config autocompletion
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'SuperDex Studio',
  tagline:
    'The GUI authoring studio for SuperDex — import, mesh, edit, and simulate bots, prefabs, and scenes.',
  url: siteUrl,
  baseUrl: siteBaseUrl,
  onBrokenLinks: 'throw',
  trailingSlash: true,
  favicon: 'img/favicon.png',
  organizationName: 'facebook',
  projectName: 'SuperDexStudio',
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  customFields: {
    fbRepoName: 'fbsource',
    ossRepoPath: '.',
    projectSuperdexUrl,
    projectSuperdexUrls,
  },

  presets: [
    [
      'docusaurus-plugin-internaldocs-fb/docusaurus-preset',
      /** @type {import('docusaurus-plugin-internaldocs-fb').PresetOptions} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        experimentalXRepoSnippets: {
          baseDir: '.',
        },
        // Must exactly match Configerator's lowercase Static Docs project key,
        // which is also the hosted URL slug.
        staticDocsProject: 'superdex_studio',
        trackingFile: 'fbcode/staticdocs/WATCHED_FILES',
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'SuperDex Studio',
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            label: 'Docs',
            position: 'left',
          },
          {
            href: projectSuperdexUrl,
            label: 'Project SuperDex',
            position: 'left',
            target: '_blank',
          },
          {
            href: 'https://github.com/facebookresearch/project_superdex',
            label: 'GitHub',
            position: 'right',
            className: 'navbar-github-button',
          },
        ],
      },
      // The footer is a swizzled component (src/theme/Footer) rather than the
      // config-driven default, so there is no `footer` key here.
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
});
