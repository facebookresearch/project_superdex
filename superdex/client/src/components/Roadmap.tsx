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
 * RELEASE ROADMAP — numbered, phased list of Project SuperDex releases.
 * Rendered on its own routed tab (pages/Roadmap.tsx).
 */
import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import type { CSSProperties } from "react";
import { docHref, type ProjectSite } from "@/lib/docs";

const ICON_BASE = `${import.meta.env.BASE_URL}img/icons/`;

type ReleaseStatus =
  | { readonly status: "Released"; readonly statusTone: "released" }
  | { readonly status: "Early Preview"; readonly statusTone: "preview" }
  | { readonly status: "Upcoming"; readonly statusTone: "upcoming" };

type Release = {
  readonly title: string;
  readonly description: string;
  readonly icon: string;
  readonly slug?: ProjectSite;
} & ReleaseStatus;

const RELEASES: readonly Release[] = [
  {
    icon: `${ICON_BASE}physics.png`,
    title: "SuperDex Physics",
    description:
      "A contact-first physics engine purpose-built for tactile manipulation, and applicable wherever stable contact and accurate sensing matter. This is the simulation backbone of Project SuperDex.",
    status: "Released",
    statusTone: "released",
    slug: "physics",
  },
  {
    icon: `${ICON_BASE}robotics.png`,
    title: "SuperDex Robotics",
    description:
      "A robotics SDK that provides robot definitions and composition, controllers, sensors, actuators, and the framework that aggregates them into complete simulation configs.",
    status: "Released",
    statusTone: "released",
    slug: "robotics",
  },
  {
    icon: `${ICON_BASE}studio.png`,
    title: "SuperDex Studio",
    description:
      "A lightweight desktop GUI authoring and visualization application for creating, editing, and validating the simulation assets — robots, meshes, task prefabs, and scenes.",
    status: "Released",
    statusTone: "released",
    slug: "studio",
  },
  {
    icon: `${ICON_BASE}lab.png`,
    title: "SuperDex Lab",
    description:
      "The simulation harness that abstracts the Markov decision process and dynamics-constrained-optimization underpinning RL, MPC, and system-ID.",
    status: "Early Preview",
    statusTone: "preview",
    slug: "lab",
  },
  {
    icon: `${ICON_BASE}teleoperation.png`,
    title: "SuperDex Teleoperation",
    description:
      "A data acquisition application that supports simulated and physical teleoperation and dataset capture.",
    status: "Upcoming",
    statusTone: "upcoming",
  },
  {
    icon: `${ICON_BASE}learning.png`,
    title: "SuperDex Learning",
    description:
      "A collection of datasets, training recipes, and evaluation suites for dexterous manipulation policy development.",
    status: "Upcoming",
    statusTone: "upcoming",
  },
];

const BORDER = "oklch(0.88 0 0)";
const NAVY = "oklch(0.28 0 0)";
const NUM = "oklch(0.6 0 0)";

const filledTag: CSSProperties = {
  background: NAVY,
  color: "oklch(0.98 0 0)",
  border: `1px solid ${NAVY}`,
};

const previewTag: CSSProperties = {
  background: "var(--preview-background)",
  color: "var(--preview-foreground)",
  border: "1px solid var(--preview-border)",
};

const upcomingTag: CSSProperties = {
  background: "oklch(0.96 0 0)",
  color: "oklch(0.35 0 0)",
  border: "1px solid oklch(0.84 0 0)",
};

const statusTagStyles: Record<ReleaseStatus["statusTone"], CSSProperties> = {
  released: filledTag,
  preview: previewTag,
  upcoming: upcomingTag,
};

export default function Roadmap() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section
      id="roadmap"
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
        <p className="section-eyebrow mb-4">Release Roadmap</p>
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
          <h1
            className="display-heading"
            style={{ fontSize: "clamp(1.8rem, 3vw, 2.6rem)" }}
          >
            Phased releases to deliver an end-to-end solution.
          </h1>
          <p className="body-text max-w-md">
            The platform ships in phases. SuperDex Physics, Robotics, Studio, and
            Lab are available today, with more to come in the coming months.
          </p>
        </div>
      </motion.div>

      {/* Numbered list */}
      <div style={{ borderTop: `1px solid ${BORDER}` }}>
        {RELEASES.map((r, i) => (
          <motion.div
            key={r.title}
            initial={{ opacity: 0, y: 16 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.45, delay: 0.1 + i * 0.06 }}
            className="grid grid-cols-[auto_auto_1fr] md:grid-cols-[auto_auto_1fr_auto] gap-x-5 sm:gap-x-8 gap-y-2 py-6 items-baseline"
            style={{ borderBottom: `1px solid ${BORDER}` }}
          >
            <span
              className="text-sm font-medium"
              style={{
                fontFamily: "'DM Sans', sans-serif",
                color: NUM,
                letterSpacing: "0.08em",
              }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <img
              src={r.icon}
              alt=""
              aria-hidden="true"
              className="w-7 h-7 self-start"
            />
            <div>
              <h2
                className="font-semibold mb-1"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.18 0 0)",
                  fontSize: "1rem",
                }}
              >
                {r.slug ? (
                  <a
                    href={docHref(r.slug)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                    style={{ color: "inherit" }}
                  >
                    {r.title}
                  </a>
                ) : (
                  r.title
                )}
              </h2>
              <p className="body-text" style={{ fontSize: "0.88rem" }}>
                {r.description}
              </p>
            </div>
            <span
              className="tag col-start-3 md:col-start-4 justify-self-start md:justify-self-end whitespace-nowrap"
              style={statusTagStyles[r.statusTone]}
            >
              {r.status}
            </span>
          </motion.div>
        ))}
        </div>
      </div>
    </section>
  );
}
