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
        description: 'Internal documentation for SuperDex Studio developers.',
      },
      items: ['internal/agents'],
    },
  ];
}

/**
 * Sidebars are defined explicitly (rather than filesystem-generated) so the
 * page order is obvious and stable. This sidebar backs the Docs navbar tab.
 */
module.exports = {
  docsSidebar: [
    'overview',
    'getting_started',
    'user_interface',
    {
      type: 'category',
      label: 'Bot Editor',
      link: {type: 'doc', id: 'bot_editor/index'},
      items: [
        'bot_editor/import',
        'bot_editor/editing',
        'bot_editor/modding',
        'bot_editor/visualization',
        'bot_editor/simulation_and_control',
        'bot_editor/archive_and_export',
      ],
    },
    'prefab_editor',
    'model_editor',
    {
      type: 'category',
      label: 'SuperDex CAD Exporter',
      link: {type: 'doc', id: 'cad_exporter/index'},
      items: [
        'cad_exporter/solidworks_tutorial',
        'cad_exporter/nx_tutorial',
        'cad_exporter/single_mesh_export',
        'cad_exporter/urdf_import',
        'cad_exporter/nx_quirks',
      ],
    },
    ...internalCategoryItems(),
  ],
};
