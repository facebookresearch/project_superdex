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
import styles from './Legend.module.css';

type Item = {
  /** The region's name, matching the label used elsewhere on the page. */
  name: React.ReactNode;
  /** One line on what the region is for. */
  text: React.ReactNode;
};

type LegendProps = {
  items: Item[];
  /**
   * Number the first item from here. Defaults to 1, which is what the callouts
   * burned into a screenshot start at.
   */
  start?: number;
};

/**
 * The key for a screenshot with numbered callouts. Numbering is derived from
 * the array order rather than written out, so the list cannot drift out of step
 * with the image the way a hand-numbered table can.
 */
export default function Legend({items, start = 1}: LegendProps) {
  return (
    <div className={styles.legend}>
      {items.map((item, i) => (
        <div className={styles.item} key={i}>
          <span className={styles.numCell}>
            <span className={styles.num} aria-hidden="true">
              {start + i}
            </span>
          </span>
          <span className={styles.name}>{item.name}</span>
          <span className={styles.text}>{item.text}</span>
        </div>
      ))}
    </div>
  );
}
