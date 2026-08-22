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

import React, { useEffect, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import useBaseUrl from '@docusaurus/useBaseUrl';
import APIFrame from '@site/src/components/api_frame';

// Fetches the manifest at runtime (not build time) so the page still renders
// an empty state when versions.json does not exist yet.
function ApiReferenceInner({ language, title }) {
  const manifestUrl = useBaseUrl('/generated/api/versions.json');
  // Unique per language so two API reference pages don't collide on the id.
  const selectId = `api-version-select-${language}`;
  const [manifest, setManifest] = useState(null);
  const [selected, setSelected] = useState(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(manifestUrl)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) {
          return;
        }
        if (data && Array.isArray(data.versions) && data.versions.length > 0) {
          setManifest(data);
          // Guard against a stale manifest.latest pointing at a missing dir.
          const latest = data.versions.includes(data.latest)
            ? data.latest
            : data.versions[0];
          setSelected(latest);
        }
        setLoaded(true);
      })
      .catch((err) => {
        // Logged so a malformed manifest is diagnosable (a dev-server SPA can
        // return 200 with index.html, making res.json() reject).
        if (!cancelled) {
          console.error(
            `Failed to load API version manifest from ${manifestUrl}:`,
            err,
          );
          setLoaded(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [manifestUrl]);

  if (!loaded) {
    return (
      <div className="api-version-bar" role="status">
        Loading API reference...
      </div>
    );
  }

  if (!manifest || !selected) {
    return (
      <div className="api-version-empty" role="status">
        API reference is unavailable for this version.
      </div>
    );
  }

  return (
    <>
      <div className="api-version-bar">
        <label htmlFor={selectId}>API Version</label>
        <select
          id={selectId}
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          {manifest.versions.map((v) => (
            <option key={v} value={v}>
              {v}
              {v === manifest.latest ? ' (latest)' : ''}
            </option>
          ))}
        </select>
      </div>
      <APIFrame
        src={`/generated/api/${selected}/${language}/index.html`}
        title={title}
      />
    </>
  );
}

export default function ApiReference({ language, title }) {
  const noticeTitleId = 'experimental-api-notice-title-' + language;
  return (
    <>
      <div
        className="alert alert--info"
        role="note"
        aria-labelledby={noticeTitleId}
      >
        <strong id={noticeTitleId}>Experimental APIs</strong>
        <div>
          Experimental APIs may be omitted from this reference. Those that are
          included may change or be removed without notice.
        </div>
      </div>
      <BrowserOnly
        fallback={
          <div className="api-version-bar" role="status">
            Loading API reference...
          </div>
        }
      >
        {() => <ApiReferenceInner language={language} title={title} />}
      </BrowserOnly>
    </>
  );
}
