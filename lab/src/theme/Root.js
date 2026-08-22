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

import Head from '@docusaurus/Head';

import ImageLightbox from './ImageLightbox';
import dmSansUrl from '../css/fonts/dm-sans-variable.woff2';
import instrumentSerifUrl from '../css/fonts/instrument-serif-regular.woff2';

export default function Root({children}) {
  return (
    <>
      <Head>
        <link
          rel="preload"
          href={dmSansUrl}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href={instrumentSerifUrl}
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
      </Head>
      {children}
      <ImageLightbox />
    </>
  );
}
