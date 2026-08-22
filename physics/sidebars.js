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
 * Parse frontmatter from a markdown file to extract sidebar_position
 */
function parseFrontmatter(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---/);

    if (!frontmatterMatch) {
      return {};
    }

    const frontmatterText = frontmatterMatch[1];
    const frontmatter = {};

    // Simple YAML parsing for sidebar_position
    const lines = frontmatterText.split('\n');
    for (const line of lines) {
      const match = line.match(/^\s*sidebar_position\s*:\s*(.+)$/);
      if (match) {
        const value = match[1].trim();
        // Parse as number if it's numeric, otherwise keep as string
        frontmatter.sidebar_position = isNaN(value) ? value : parseInt(value, 10);
        break;
      }
    }

    return frontmatter;
  } catch (error) {
    console.warn(`Warning: Could not parse frontmatter from ${filePath}:`, error.message);
    return {};
  }
}

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

/**
 * Converts snake_case to Pretty Case, falls back to original name for non-snake_case
 * Examples:
 * - user_docs -> User Docs (true snake_case conversion)
 * - mochi_core -> Mochi Core (true snake_case conversion)
 */
function snakeCaseToPretty(str) {
  // Regex to match true snake_case: lowercase letters/numbers, separated by single underscores
  // ^[a-z0-9]+(_[a-z0-9]+)*$ means:
  // - ^ start of string
  // - [a-z0-9]+ one or more lowercase letters or numbers
  // - (_[a-z0-9]+)* zero or more groups of (underscore followed by lowercase letters/numbers)
  // - $ end of string
  const snakeCaseRegex = /^[a-z0-9]+(_[a-z0-9]+)*$/;

  // Only convert if it matches true snake_case pattern
  if (snakeCaseRegex.test(str)) {
    return str
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  // For non-snake_case names, return as-is to avoid breaking the site
  // This allows graceful degradation - the name will display but won't be "pretty"
  return str;
}

/**
 * Recursively generates sidebar items with pretty names for directories
 */
function generateSidebarItems(dirPath, basePath = '') {
  const items = [];

  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });

    // Separate directories and files for different sorting logic
    const directories = [];
    const files = [];

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      const relativePath = basePath ? `${basePath}/${entry.name}` : entry.name;

      if (entry.isDirectory()) {
        // Skip hidden directories and node_modules
        if (entry.name.startsWith('.') || entry.name === 'node_modules') {
          continue;
        }

        const subItems = generateSidebarItems(fullPath, relativePath);

        if (subItems.length > 0) {
          directories.push({
            type: 'category',
            label: snakeCaseToPretty(entry.name),
            items: subItems,
            link: {
              type: 'generated-index',
              description: `Documentation for ${snakeCaseToPretty(entry.name)}.`
            },
            _sortName: entry.name
          });
        }
      } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
        // Skip certain files
        if (entry.name === 'README.md' || entry.name.startsWith('_')) {
          continue;
        }

        // Parse frontmatter to get sidebar_position
        const frontmatter = parseFrontmatter(fullPath);

        // Strip numeric prefix from filename (e.g., 0_getting_started.md -> getting_started)
        // This matches Docusaurus's default behavior of removing number prefixes
        let fileName = entry.name.replace(/\.(md|mdx)$/, '');
        fileName = fileName.replace(/^\d+_/, ''); // Remove leading digits and underscore

        const docId = basePath ? `${basePath}/${fileName}` : fileName;

        files.push({
          docId,
          name: entry.name,
          sidebar_position: frontmatter.sidebar_position
        });
      }
    }

    // Read position and label from _category_.json for directories
    directories.forEach(dir => {
      const categoryPath = path.join(dirPath, dir._sortName, '_category_.json');
      try {
        if (fs.existsSync(categoryPath)) {
          const categoryContent = fs.readFileSync(categoryPath, 'utf8');
          const categoryData = JSON.parse(categoryContent);
          if (categoryData.position !== undefined) {
            dir._position = categoryData.position;
          }
          if (categoryData.label !== undefined) {
            dir.label = categoryData.label;
          }
          if (categoryData.link !== undefined) {
            dir.link = categoryData.link;
          }
        }
      } catch (error) {
        // Ignore errors, directory will sort alphabetically
      }
    });

    // Create unified array of all items with positions
    const allItems = [
      ...directories.map(dir => ({
        type: 'directory',
        position: dir._position,
        name: dir._sortName,
        item: dir
      })),
      ...files.map(file => ({
        type: 'file',
        position: file.sidebar_position,
        name: file.name,
        item: file
      }))
    ];

    // Sort all items by position, then alphabetically
    allItems.sort((a, b) => {
      // If both have position, sort by that
      if (a.position !== undefined && b.position !== undefined) {
        return a.position - b.position;
      }
      // If only one has position, it comes first
      if (a.position !== undefined && b.position === undefined) {
        return -1;
      }
      if (a.position === undefined && b.position !== undefined) {
        return 1;
      }
      // If neither has position, sort alphabetically
      return a.name.localeCompare(b.name);
    });

    // Add sorted items to result
    allItems.forEach(({ type, item }) => {
      if (type === 'directory') {
        const { _sortName, _position, ...cleanDir } = item;
        items.push(cleanDir);
      } else {
        items.push(item.docId);
      }
    });

  } catch (error) {
    console.warn(`Warning: Could not read directory ${dirPath}:`, error.message);
  }

  return items;
}

// Generate per-section sidebars
const docsPath = path.join(__dirname, 'docs');
const internalDocsPath = path.join(docsPath, 'internal');

module.exports = {
  docsSidebar: [
    'overview',
    'getting_started',
    {
      type: 'category',
      label: 'Concepts',
      items: generateSidebarItems(
        path.join(docsPath, 'concepts'),
        'concepts',
      ),
      link: {
        type: 'generated-index',
        description:
          'Core concepts of the SuperDex Physics simulation library.',
      },
    },
    {
      type: 'category',
      label: 'Authoring Scenes',
      items: generateSidebarItems(
        path.join(docsPath, 'authoring_scenes'),
        'authoring_scenes',
      ),
      link: {
        type: 'generated-index',
        description:
          'Prepare simulation-ready model assets with SuperDex Physics, then compose actors and settings into reusable prefabs.',
      },
    },
    {
      type: 'doc',
      id: 'debugging_scenes',
      label: 'Inspecting Scenes',
    },
    // docs/internal/ is stripped from the open-source export, so guard the whole
    // category: without this the public build renders an empty navigation entry
    // labelled "Internal" whose landing page lists nothing.
    ...(fs.existsSync(internalDocsPath)
      ? [
          {
            type: 'category',
            label: 'Internal',
            items: generateSidebarItems(internalDocsPath, 'internal'),
            link: {
              type: 'generated-index',
              description:
                'Internal documentation for SuperDex Physics developers.',
            },
          },
        ]
      : []),
  ],
  examplesSidebar: generateSidebarItems(
    path.join(docsPath, 'examples'),
    'examples',
  ),
  apiSidebar: generateSidebarItems(
    path.join(docsPath, 'api_reference'),
    'api_reference',
  ),
};
