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

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const googleTagId = process.env.SUPERDEX_GOOGLE_TAG_ID?.trim() ?? "";
const isPublicBuild = process.env.VITE_PUBLIC_BUILD === "1";
const publicBaseUrl = process.env.SUPERDEX_PUBLIC_BASE_URL;
const staticDocsBaseUrl = process.env.STATIC_DOCS_BASE_URL;
const enableGoogleTag = isPublicBuild && Boolean(googleTagId);

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: "superdex-google-tag",
      transformIndexHtml() {
        if (!enableGoogleTag) {
          return [];
        }

        return [
          {
            tag: "script",
            attrs: {
              async: true,
              src: `https://www.googletagmanager.com/gtag/js?id=${googleTagId}`,
            },
            injectTo: "head-prepend",
          },
          {
            tag: "script",
            children: `
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', '${googleTagId}', {
  anonymize_ip: true,
  send_page_view: false,
});
            `.trim(),
            injectTo: "head-prepend",
          },
        ];
      },
    },
  ],
  base: publicBaseUrl || staticDocsBaseUrl || "/",
  define: {
    __SUPERDEX_INTERNAL_STATIC_DOCS__: JSON.stringify(
      Boolean(staticDocsBaseUrl) && !publicBaseUrl,
    ),
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "client", "src"),
    },
  },
  root: path.resolve(__dirname, "client"),
  build: {
    outDir: path.resolve(__dirname, "dist/public"),
    emptyOutDir: true,
  },
});
