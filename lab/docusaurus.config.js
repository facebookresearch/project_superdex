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

const lightCodeTheme = {
  plain: {
    backgroundColor: '#f6f8fa',
    color: '#1f2328',
  },
  styles: [
    {
      types: ['comment', 'prolog', 'doctype', 'cdata'],
      style: {color: '#57606a', fontStyle: 'italic'},
    },
    {
      types: ['punctuation', 'operator'],
      style: {color: '#1f2328'},
    },
    {
      types: ['property', 'boolean', 'number', 'constant', 'symbol'],
      style: {color: '#0550ae'},
    },
    {
      types: ['selector', 'attr-name', 'string', 'char', 'builtin'],
      style: {color: '#0a3069'},
    },
    {
      types: ['atrule', 'attr-value', 'keyword', 'deleted'],
      style: {color: '#cf222e'},
    },
    {
      types: ['function', 'class-name', 'tag'],
      style: {color: '#8250df'},
    },
    {
      types: ['regex', 'important', 'variable'],
      style: {color: '#0550ae'},
    },
    {
      types: ['inserted'],
      style: {color: '#116329'},
    },
  ],
};
const darkCodeTheme = {
  plain: {
    backgroundColor: '#0d1117',
    color: '#e6edf3',
  },
  styles: [
    {
      types: ['comment', 'prolog', 'doctype', 'cdata'],
      style: {color: '#8b949e'},
    },
    {
      types: ['punctuation', 'operator'],
      style: {color: '#e6edf3'},
    },
    {
      types: ['property', 'boolean', 'number', 'constant', 'symbol'],
      style: {color: '#79c0ff'},
    },
    {
      types: ['number', 'boolean', 'null'],
      languages: ['json', 'webmanifest'],
      style: {color: '#ffa657'},
    },
    {
      types: ['selector', 'attr-name', 'string', 'char', 'builtin'],
      style: {color: '#a5d6ff'},
    },
    {
      types: ['atrule', 'attr-value', 'keyword', 'deleted'],
      style: {color: '#ff7b72'},
    },
    {
      types: ['function', 'class-name', 'tag'],
      style: {color: '#d2a8ff'},
    },
    {
      types: ['regex', 'important', 'variable'],
      style: {color: '#ffa657'},
    },
    {
      types: ['inserted'],
      style: {color: '#7ee787'},
    },
  ],
};

const isPublicBuild = process.env.SUPERDEX_PUBLIC_BUILD === '1';
const googleTagId = (process.env.SUPERDEX_GOOGLE_TAG_ID || '').trim();

// A public build must run from the OSS export, where ShipIt has already stripped
// docs/internal/. If that directory is present the export has not happened, so
// refuse rather than emit internal pages: the internaldocs preset copies docs/
// verbatim to build/_src/ and no docs `exclude` can stop that copy. Only `build`
// and `deploy` emit output; `start` is exempt so the README's public-URL preview
// still works.
const docusaurusSubcommand = process.argv.slice(2).find((a) => !a.startsWith('-'));
if (
  isPublicBuild &&
  (docusaurusSubcommand === 'build' || docusaurusSubcommand === 'deploy') &&
  require('fs').existsSync(require('path').join(__dirname, 'docs', 'internal'))
) {
  throw new Error(
    'SUPERDEX_PUBLIC_BUILD=1 but docs/internal/ exists. A public build must run ' +
      'from the exported tree; building here would publish internal pages and ' +
      'their raw markdown under _src/internal/.',
  );
}
const publicOrigin = (
  process.env.SUPERDEX_PUBLIC_ORIGIN || 'https://projectsuperdex.com'
).replace(/\/+$/, '');
const publicBasePath = (
  process.env.SUPERDEX_PUBLIC_BASE_URL || '/'
).replace(/^\/+|\/+$/g, '');
const projectBaseUrl = publicBasePath ? `/${publicBasePath}/` : '/';
const siteUrl = isPublicBuild ? publicOrigin : 'https://internalfb.com';
const siteBaseUrl = isPublicBuild ? `${projectBaseUrl}lab/` : '/';
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
  title: 'SuperDex Lab',
  tagline:
    'The research and experimentation hub for SuperDex — RL environments, training, and benchmarking.',
  url: siteUrl,
  baseUrl: siteBaseUrl,
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  trailingSlash: true,
  favicon: 'img/favicon.png',
  organizationName: 'facebookresearch',
  projectName: 'project_superdex',
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  customFields: {
    // `fbRepoName` is only read by the internal code-block frame, which is inert
    // on a public host, so the value is withheld from a public build. The
    // preset's frame *code* still ships; that is preset behaviour this config
    // does not control.
    ...(isPublicBuild ? {} : {fbRepoName: 'fbsource'}),
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
        // Must exactly match the lowercase Static Docs project key, which is
        // also the hosted URL slug.
        staticDocsProject: 'superdex_lab',
        // Watched-files tracking runs only in the internal build.
        ...(isPublicBuild
          ? {}
          : {trackingFile: 'fbcode/staticdocs/WATCHED_FILES'}),
        blog: false,
        gtag:
          isPublicBuild && googleTagId
            ? {trackingID: googleTagId, anonymizeIP: true}
            : undefined,
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
        title: 'SuperDex Lab',
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
