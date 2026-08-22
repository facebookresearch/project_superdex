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

const fs = require('fs');
const path = require('path');

/**
 * Only add this category when its directory exists: naming a doc id with no
 * backing file fails checkSidebarsDocIds. Guarding on the directory rather than
 * on the page keeps a renamed or retitled page failing loudly. A doc id defaults
 * to its path-relative filename, and is overridden by an `id:` frontmatter field
 * — which is why `AGENTS.md` is referenced as `internal/agents`.
 */
function internalCategoryItems() {
  if (!fs.existsSync(path.join(__dirname, 'docs', 'internal'))) {
    return [];
  }
  return [
    {
      type: 'category',
      label: 'Internal',
      link: {
        type: 'generated-index',
        description: 'Internal documentation for SuperDex Lab developers.',
      },
      items: [
        'internal/agents',
        {
          type: 'category',
          label: 'SuperDex Gym (internal)',
          items: [
            'internal/superdex_gym/internal_environments',
            'internal/superdex_gym/setup_meta',
            'internal/superdex_gym/benchmarking_meta',
          ],
        },
      ],
    },
  ];
}

/**
 * Sidebars are defined explicitly (rather than filesystem-generated) so the
 * page order is obvious and stable. A single sidebar backs the Docs tab.
 *
 * SuperDex Lab is the umbrella for SuperDex's research & experimentation
 * tooling. All three components — the environments, the RLlib training scripts
 * and the benchmarking tools — are documented under the SuperDex Gym category
 * for now; additional components get their own categories as they land.
 */
module.exports = {
  docsSidebar: [
    'overview',
    {
      type: 'category',
      label: 'SuperDex Gym',
      collapsed: false,
      link: {
        type: 'generated-index',
        description:
          'SuperDex Physics implementation of the Gymnasium interface — a reinforcement-learning framework for training agent controllers.',
      },
      items: [
        'superdex_gym/intro',
        'superdex_gym/setup',
        'superdex_gym/running_examples',
        'superdex_gym/examples',
        'superdex_gym/env_reference',
        'superdex_gym/benchmarking',
        'superdex_gym/environments',
        'superdex_gym/batching',
        'superdex_gym/rllib',
        'superdex_gym/render',
        'superdex_gym/visualize_training_history',
      ],
    },
    ...internalCategoryItems(),
  ],
};
