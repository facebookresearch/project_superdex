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
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

// Links map to this site's own tabs (validated against the live navbar routes).
const QUICK_LINKS = [
  {label: 'Docs', to: '/docs/overview/'},
  {label: 'Examples', to: '/docs/examples/getting_started/'},
  {label: 'API Reference', to: '/docs/api_reference/cpp/'},
];

const EXPLORE_LINKS = [
  {label: 'SuperDex Physics', site: 'physics'},
  {label: 'SuperDex Robotics', site: 'robotics'},
  {label: 'SuperDex Studio', site: 'studio'},
  {label: 'SuperDex Lab', site: 'lab'},
];

export default function Footer() {
  const {siteConfig} = useDocusaurusContext();
  const {projectSuperdexUrls} = siteConfig.customFields;
  const resources = [
    {
      label: 'GitHub',
      href: 'https://github.com/facebookresearch/project_superdex',
    },
    {
      label: 'Project SuperDex',
      href: siteConfig.customFields.projectSuperdexUrl,
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    {
      label: 'Terms of Use',
      href: 'https://opensource.fb.com/legal/terms',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    {
      label: 'Privacy Policy',
      href: 'https://opensource.fb.com/legal/privacy',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
  ];
  const year = new Date().getFullYear();
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.columns}>
          <div className={styles.brandCol}>
            <div className={styles.brandName}>SuperDex Physics</div>
            <p className={styles.brandDesc}>
              A contact-rich physics engine for tactile manipulation. Built by
              Meta Reality Labs Research.
            </p>
          </div>

          <div className={styles.linkCol}>
            <p className={styles.colTitle}>Quick Links</p>
            {QUICK_LINKS.map((l) => (
              <Link key={l.label} to={l.to} className={styles.link}>
                {l.label}
              </Link>
            ))}
          </div>

          <div className={styles.linkCol}>
            <p className={styles.colTitle}>Explore SuperDex</p>
            {EXPLORE_LINKS.map((l) => (
              <Link
                key={l.site}
                href={projectSuperdexUrls[l.site]}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.link}>
                {l.label}
              </Link>
            ))}
          </div>

          <div className={styles.linkCol}>
            <p className={styles.colTitle}>Resources</p>
            {resources.map((l) => (
              <Link
                key={l.label}
                href={l.href}
                target={l.target}
                rel={l.rel}
                className={styles.link}>
                {l.label}
              </Link>
            ))}
          </div>
        </div>

        <div className={styles.bottomBar}>
          <p className={styles.copyright}>
            {`Copyright © ${year} Meta Platforms, Inc. All rights reserved.`}
          </p>
          <p className={styles.credit}>Built by Meta Reality Labs Research</p>
        </div>
      </div>
    </footer>
  );
}
