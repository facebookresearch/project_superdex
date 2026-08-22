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

/*
 * Link helper for the per-layer docs sites.
 *
 * Public builds publish each layer beneath the configured GitHub
 * Pages project root. Internal builds publish them as independent Static Docs
 * projects, so their canonical URLs cannot be derived from the landing base.
 */
const INTERNAL_DOC_URLS = {
  // @oss-disable: physics: "https://www.internalfb.com/intern/staticdocs/mochi_physics/",
  physics: `${import.meta.env.BASE_URL}physics/`, // @oss-enable
  // @oss-disable: robotics: "https://www.internalfb.com/intern/staticdocs/superdex_robotics/",
  robotics: `${import.meta.env.BASE_URL}robotics/`, // @oss-enable
  // @oss-disable: lab: "https://www.internalfb.com/intern/staticdocs/superdex_lab/",
  lab: `${import.meta.env.BASE_URL}lab/`, // @oss-enable
  // @oss-disable: studio: "https://www.internalfb.com/intern/staticdocs/superdex_studio/",
  studio: `${import.meta.env.BASE_URL}studio/`, // @oss-enable
} as const;

export type ProjectSite = keyof typeof INTERNAL_DOC_URLS;

declare const __SUPERDEX_INTERNAL_STATIC_DOCS__: boolean;

export function docHref(slug: ProjectSite): string {
  return __SUPERDEX_INTERNAL_STATIC_DOCS__
    ? INTERNAL_DOC_URLS[slug]
    : `${import.meta.env.BASE_URL}${slug}/`;
}
