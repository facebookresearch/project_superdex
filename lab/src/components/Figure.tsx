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

import React, {useEffect, useState} from 'react';
import styles from './Figure.module.css';

type Variant = 'full' | 'inset' | 'strip' | 'float';

type FigureProps = {
  src: string;
  /** Describes the image for readers who can't see it. Never duplicate the caption here. */
  alt: string;
  /** Visible supplementary line. Optional — not every figure needs one. */
  caption?: React.ReactNode;
  variant?: Variant;
  /**
   * Hard cap on rendered width, in px. Several captures are small by nature
   * (the context menus are ~220px wide); scaling those up only blurs them.
   */
  maxWidth?: number;
  /**
   * Cap the figure's height in px and let it scroll instead. For a tall docked
   * panel, where scaling the whole thing down to fit would make its interface
   * text unreadable.
   */
  maxHeight?: number;
};

/**
 * A float is sized by the element that is actually floated, so its cap has to
 * land on the <figure>. Every other variant sizes the inner wrapper instead.
 */
function widthStyle(variant: Variant, maxWidth?: number) {
  if (!maxWidth) return {};
  return variant === 'float' ? {outer: {maxWidth}} : {inner: {maxWidth}};
}

export default function Figure({
  src,
  alt,
  caption,
  variant = 'full',
  maxWidth,
  maxHeight,
}: FigureProps) {
  const {outer, inner} = widthStyle(variant, maxWidth);
  return (
    <figure className={`${styles.figure} ${styles[variant] ?? ''}`} style={outer}>
      {maxHeight ? (
        // The fade that signals "more below" has to sit outside the scroller,
        // or it would scroll away with the content it is advertising.
        <div className={styles.scrollWrap} style={inner}>
          <div className={`${styles.frame} ${styles.scroller}`} style={{maxHeight}}>
            <img className={styles.image} src={src} alt={alt} loading="lazy" />
          </div>
        </div>
      ) : (
        <div className={styles.frame} style={inner}>
          <img className={styles.image} src={src} alt={alt} loading="lazy" />
        </div>
      )}
      {caption && (
        <figcaption className={styles.caption} style={inner}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

type FigureRowProps = {
  children: React.ReactNode;
  cols?: 2 | 3;
  /**
   * Narrowest a column may get, in px, before the row folds to a single
   * column. Use for 1x captures, which cannot be scaled down without their
   * interface text going soft. Overrides `cols`.
   */
  minWidth?: number;
};

/** An n-up comparison. Works best when the children are exported at matching dimensions. */
export function FigureRow({children, cols = 3, minWidth}: FigureRowProps) {
  if (minWidth) {
    return (
      <div
        className={`${styles.row} ${styles.rowFit}`}
        style={{['--sdx-row-min' as string]: `${minWidth}px`}}>
        {children}
      </div>
    );
  }
  return (
    <div className={`${styles.row} ${cols === 2 ? styles.row2 : styles.row3}`}>
      {children}
    </div>
  );
}

/** Bolds the leading term of a comparison caption, e.g. the view-mode name. */
export function FigureLabel({children}: {children: React.ReactNode}) {
  return <span className={styles.rowLabel}>{children}</span>;
}

/**
 * A caption-styled footnote for an asterisked disclaimer under a figure or a
 * video, where the note qualifies the media rather than describing it. Set
 * `standalone` to qualify a passage of prose instead — an aside that would
 * overstate itself as a heading.
 */
export function FigureNote({
  children,
  standalone,
}: {
  children: React.ReactNode;
  standalone?: boolean;
}) {
  return (
    <p className={`${styles.note} ${standalone ? styles.noteStandalone : ''}`}>
      {children}
    </p>
  );
}

type FigureVideoProps = {
  src: string;
  poster: string;
  alt: string;
  caption?: React.ReactNode;
  variant?: Variant;
  maxWidth?: number;
};

/**
 * A short silent clip used as an animated diagram: it loops on its own and
 * shows no chrome. Under `prefers-reduced-motion` it degrades to the poster
 * frame plus real controls, so the reader opts in to the motion.
 */
export function FigureVideo({
  src,
  poster,
  alt,
  caption,
  variant = 'full',
  maxWidth,
}: FigureVideoProps) {
  // Resolved after mount: the server render has no media query to consult, and
  // guessing wrong would ship an autoplaying video to someone who asked for none.
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => setReduceMotion(mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  }, []);

  const {outer, inner} = widthStyle(variant, maxWidth);
  return (
    <figure className={`${styles.figure} ${styles[variant] ?? ''}`} style={outer}>
      <div className={styles.frame} style={inner}>
        <video
          className={styles.video}
          src={src}
          poster={poster}
          aria-label={alt}
          autoPlay={!reduceMotion}
          loop={!reduceMotion}
          controls={reduceMotion}
          muted
          playsInline
          preload="metadata"
        />
      </div>
      {caption && (
        <figcaption className={styles.caption} style={inner}>
          {caption}
        </figcaption>
      )}
    </figure>
  );
}
