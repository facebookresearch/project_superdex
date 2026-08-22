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

import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

export default function ProjectSuperdexLink({site, children, ...props}) {
  const {siteConfig} = useDocusaurusContext();
  const href = siteConfig.customFields.projectSuperdexUrls[site];
  if (!href) {
    throw new Error(`Unknown Project SuperDex site: ${site}`);
  }
  return (
    <a
      {...props}
      href={href}
      target="_blank"
      rel="noopener noreferrer">
      {children}
    </a>
  );
}
