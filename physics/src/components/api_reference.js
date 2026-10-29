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
import BrowserOnly from '@docusaurus/BrowserOnly';
import APIFrame from '@site/src/components/api_frame';

function ApiReferenceStatus({ children }) {
  return (
    <div className="api-reference-status" role="status">
      {children}
    </div>
  );
}

// Current documentation passes apiVersion="latest"; numbered documentation
// snapshots pass the concrete artifact selected when they were created.
function ApiReferenceInner({ apiVersion, language, title }) {
  if (!apiVersion) {
    return (
      <ApiReferenceStatus>
        API reference is unavailable for this version.
      </ApiReferenceStatus>
    );
  }
  return (
    <APIFrame
      src={`/generated/api/${apiVersion}/${language}/index.html`}
      title={title}
    />
  );
}

export default function ApiReference({ apiVersion, language, title }) {
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
          <ApiReferenceStatus>Loading API reference...</ApiReferenceStatus>
        }
      >
        {() => (
          <ApiReferenceInner
            apiVersion={apiVersion}
            language={language}
            title={title}
          />
        )}
      </BrowserOnly>
    </>
  );
}
