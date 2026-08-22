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
 * GET STARTED PAGE CONTENT — a short build-from-source blurb that links to the
 * canonical README on GitHub, followed by the layer entry points
 * (Physics, Robotics, Studio, Lab).
 * Editorial layout on white; the cards link to each layer's documentation.
 *
 * The build/run steps intentionally live ONLY in the repo README
 * (README.md at the repository root) so the commands never drift between the
 * site and the docs. Update the README, not this file, to change them.
 *
 * The layer-card content itself lives in lib/layers.ts. Edit it there, not
 * here.
 */
import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import { ArrowRight, BookOpen } from "lucide-react";
import { docHref } from "@/lib/docs";
import { LAYERS } from "@/lib/layers";

// The repo README (rendered on the GitHub project home page) is the canonical
// build/getting-started guide.
const README_URL = "https://github.com/facebookresearch/project_superdex";

export default function GetStarted() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section
      id="get-started"
      className="py-12 lg:py-16"
      style={{ background: "oklch(1 0 0)" }}
    >
      <div className="container" ref={ref}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.55 }}
          className="mb-14"
        >
          <p className="section-eyebrow mb-4">Get Started</p>
          <h1
            className="display-heading"
            style={{ fontSize: "clamp(2rem, 3.5vw, 3rem)" }}
          >
            Build with SuperDex today.
          </h1>
        </motion.div>

        {/* Build & run — short blurb pointing at the canonical README guide */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.55, delay: 0.05 }}
          className="mb-16"
        >
          <p className="body-text max-w-2xl mb-6" style={{ fontSize: "0.95rem" }}>
            SuperDex builds from source with CMake — clone the repository,
            configure, and build. The full step-by-step guide (prerequisites,
            build flags, and how to run the robotics examples) lives in the
            README.
          </p>
          <a
            href={README_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 rounded text-sm font-medium no-underline transition-all duration-200"
            style={{
              fontFamily: "'DM Sans', sans-serif",
              background: "oklch(0.18 0 0)",
              color: "oklch(0.98 0 0)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "oklch(0.28 0 0)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "oklch(0.18 0 0)")
            }
          >
            <BookOpen aria-hidden="true" size={15} />
            Read the build guide on GitHub
            <ArrowRight aria-hidden="true" size={14} />
          </a>
          <p
            className="text-xs mt-5"
            style={{
              fontFamily: "'DM Sans', sans-serif",
              color: "oklch(0.45 0 0)",
            }}
          >
            Prerequisites: CMake 3.25+ · Clang on Linux/macOS or ClangCL on
            Windows · Git
          </p>
        </motion.div>

        <hr className="divider mb-14" />

        {/* Cards */}
        <p className="section-eyebrow mb-3">Explore the building blocks</p>
        <p className="body-text mb-8 lg:whitespace-nowrap">
          Click on the Project SuperDex building blocks below to explore documentation and API references where available.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-5">
          {LAYERS.map((layer, i) => {
            return (
              <motion.a
                key={layer.title}
                aria-label={`Get started with ${layer.title}`}
                href={docHref(layer.slug)}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 24 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: i * 0.08 }}
                className="editorial-card no-underline flex flex-col p-5"
              >
                <div
                  className="w-14 h-14 rounded-lg flex items-center justify-center mb-4"
                  style={{ background: "oklch(0.93 0 0)" }}
                >
                  <img
                    src={layer.icon}
                    alt=""
                    aria-hidden="true"
                    className="w-10 h-10"
                  />
                </div>
                <h2
                  className="font-semibold mb-2"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.18 0 0)",
                    fontSize: "0.95rem",
                  }}
                >
                  {layer.title}
                </h2>
                <p
                  className="body-text mb-5 flex-1"
                  style={{ fontSize: "0.82rem" }}
                >
                  {layer.blurb}
                </p>
                <span className="link-arrow" style={{ fontSize: "0.8rem" }}>
                  Get started <ArrowRight aria-hidden="true" size={11} />
                </span>
              </motion.a>
            );
          })}
        </div>
      </div>
    </section>
  );
}
