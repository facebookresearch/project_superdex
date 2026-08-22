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

/**
 * Site-wide image lightbox. Any image in the doc body opens full-screen on
 * click (or Enter/Space); Esc, the backdrop, the image, or the close button
 * dismisses it. Works for both the <Figure> component and hand-rolled <img>,
 * because it hooks the rendered DOM rather than any one component.
 */

import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import {useLocation} from '@docusaurus/router';
import styles from './ImageLightbox.module.css';

// The container Docusaurus wraps doc content in. Scoping to it excludes the
// navbar logo, footer, and admonition icons.
const CONTENT_SELECTOR = '.theme-doc-markdown';

// A content image that isn't a link (a click on a linked image should follow
// the link, not zoom), isn't opted out (an inline UI icon, or anything under
// [data-no-lightbox]), and isn't the enlarged image inside the overlay itself.
function isZoomable(img) {
  if (!img || img.tagName !== 'IMG') return false;
  if (!img.closest(CONTENT_SELECTOR)) return false;
  if (img.closest('a')) return false;
  if (img.closest('[data-no-lightbox], .sdx-inline-icon')) return false;
  return true;
}

export default function ImageLightbox() {
  const [active, setActive] = useState(null); // {src, alt} | null
  const [mounted, setMounted] = useState(false);
  const dialogRef = useRef(null);
  const triggerRef = useRef(null);
  const location = useLocation();

  useEffect(() => setMounted(true), []);

  const open = useCallback((img) => {
    triggerRef.current = img;
    setActive({src: img.currentSrc || img.src, alt: img.getAttribute('alt') || ''});
  }, []);

  const close = useCallback(() => setActive(null), []);

  // Root persists across SPA navigation, so a Back/Forward taken while the
  // overlay is open would strand the old image over the new page and leave
  // triggerRef pointing at a detached node. Close and forget on any route
  // change (pathname, search, or hash).
  useEffect(() => {
    setActive(null);
    triggerRef.current = null;
  }, [location.pathname, location.search, location.hash]);

  // Make eligible images operable by keyboard/AT after each navigation. Zoom
  // itself works via delegation regardless; this adds focusability + labels.
  // Uses the same isZoomable() predicate as activation, so an opted-out image
  // (inline icon, [data-no-lightbox]) never becomes a keyboard tab stop either.
  useEffect(() => {
    const root = document.querySelector(CONTENT_SELECTOR);
    if (!root) return;
    root.querySelectorAll('img').forEach((img) => {
      if (!isZoomable(img)) return;
      img.classList.add(styles.zoomable);
      img.setAttribute('tabindex', '0');
      img.setAttribute('role', 'button');
      const alt = img.getAttribute('alt');
      img.setAttribute('aria-label', alt ? `Expand image: ${alt}` : 'Expand image');
    });
  }, [location.pathname]);

  // One delegated listener each for pointer and keyboard, so images added by
  // client-side navigation stay covered with no re-wiring.
  useEffect(() => {
    const onClick = (e) => {
      const img = e.target.closest?.('img');
      if (isZoomable(img)) {
        e.preventDefault();
        open(img);
      }
    };
    const onKeyDown = (e) => {
      if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
      const img = e.target.closest?.('img');
      if (isZoomable(img)) {
        e.preventDefault();
        open(img);
      }
    };
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('click', onClick);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  // Drive the native <dialog>: showModal() gives a focus trap and Esc-to-close
  // for free. onClose (fired by Esc too) restores focus to the source image.
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (active && !dialog.open) {
      dialog.showModal();
    } else if (!active && dialog.open) {
      dialog.close();
    }
  }, [active]);

  if (!mounted) return null;

  const handleClose = () => {
    close();
    triggerRef.current?.focus?.();
  };

  return createPortal(
    <dialog
      ref={dialogRef}
      className={styles.dialog}
      aria-label={active?.alt ? `Expanded image: ${active.alt}` : 'Expanded image'}
      onClose={handleClose}
      onClick={handleClose}>
      {active && (
        <>
          <button
            type="button"
            className={styles.close}
            aria-label="Close"
            onClick={handleClose}>
            &times;
          </button>
          <img className={styles.image} src={active.src} alt={active.alt} />
        </>
      )}
    </dialog>,
    document.body,
  );
}
