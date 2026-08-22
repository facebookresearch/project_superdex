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
 * LAYERS — content for the four layer entry points on Get Started.
 */
import type { ProjectSite } from "@/lib/docs";

const ICON_BASE = `${import.meta.env.BASE_URL}img/icons/`;

export type Layer = {
  /**
   * Reuses `ProjectSite` (derived from the URL map in lib/docs.ts) rather than
   * re-declaring the slug literals here — every consumer passes this straight
   * to docHref(), so there should only ever be one list of slugs.
   */
  slug: ProjectSite;
  title: string;
  icon: string;
  blurb: string;
};

export const LAYERS: readonly Layer[] = [
  {
    slug: "physics",
    title: "SuperDex Physics",
    icon: `${ICON_BASE}physics.png`,
    blurb:
      "A contact-first physics engine purpose-built for tactile manipulation, and applicable wherever stable contact and accurate sensing matter. This is the simulation backbone of Project SuperDex.",
  },
  {
    slug: "robotics",
    title: "SuperDex Robotics",
    icon: `${ICON_BASE}robotics.png`,
    blurb:
      "A robotics SDK that provides robot definitions and composition, controllers, sensors, actuators, and the framework that aggregates them into complete simulation configs.",
  },
  {
    slug: "studio",
    title: "SuperDex Studio",
    icon: `${ICON_BASE}studio.png`,
    blurb:
      "A lightweight desktop GUI authoring and visualization application for creating, editing, and validating the simulation assets — robots, meshes, task prefabs, and scenes.",
  },
  {
    slug: "lab",
    title: "SuperDex Lab",
    icon: `${ICON_BASE}lab.png`,
    blurb:
      "The simulation harness that abstracts the Markov decision process and dynamics-constrained-optimization underpinning RL, MPC, and system-ID.",
  },
] as const;
