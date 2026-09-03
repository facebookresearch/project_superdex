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
 * on the page keeps a renamed or retitled page failing loudly. Doc ids come from
 * the `id:` frontmatter field, not the filename.
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
        description: 'Internal documentation for SuperDex Robotics developers.',
      },
      items: ['internal/agents'],
    },
  ];
}

/**
 * Sidebars are defined explicitly (rather than filesystem-generated) so the
 * page order is obvious and stable. Two independent sidebars back the two
 * navbar tabs: Docs and Examples.
 */
module.exports = {
  docsSidebar: [
    'overview',
    'getting_started',
    'bots',
    'modifying_bots',
    'bot_context_lifetime',
    'bot_components',
    'bot_assets',
    ...internalCategoryItems(),
  ],
  examplesSidebar: [
    'examples/overview',
    {
      type: 'category',
      label: 'Basic',
      items: [
        'examples/basic/loading_a_bot',
        'examples/basic/loading_a_scene',
        'examples/basic/loading_from_urdf',
      ],
    },
    {
      type: 'category',
      label: 'Control',
      items: [
        'examples/control/jsc_control',
        'examples/control/osc_control',
        'examples/control/ik_pose_control',
        'examples/control/osc_jsc_control',
        'examples/control/bimanual_control',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: ['examples/advanced/custom_components'],
    },
  ],
  apiSidebar: [
    'api_reference/cpp',
    // 'api_reference/c',
    'api_reference/python',
  ],
};
