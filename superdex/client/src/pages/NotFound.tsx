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

import Navbar from "@/components/Navbar";
import { Link } from "wouter";

const NAV_HEIGHT = 64;

export default function NotFound() {
  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "oklch(1 0 0)" }}
    >
      <Navbar />
      <div style={{ height: NAV_HEIGHT }} />

      <main>
        <section className="py-24 flex-1 flex items-center">
        <div className="container">
          <p className="section-eyebrow mb-4">404</p>
          <h1
            className="display-heading mb-6"
            style={{ fontSize: "clamp(2rem, 4vw, 3.5rem)" }}
          >
            Page not found.
          </h1>
          <p className="body-text max-w-xl mb-4">
            Sorry, the page you are looking for doesn't exist.
          </p>
          <p className="body-text max-w-xl mb-10">
            It may have been moved or deleted.
          </p>

          <Link
            href="/"
            className="inline-block px-5 py-2.5 rounded text-[0.82rem] font-medium no-underline transition-colors duration-200"
            style={{
              background: "oklch(0.15 0 0)",
              color: "oklch(0.97 0 0)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "oklch(0.25 0 0)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "oklch(0.15 0 0)")
            }
          >
            Go home
          </Link>
        </div>
        </section>
      </main>
    </div>
  );
}
