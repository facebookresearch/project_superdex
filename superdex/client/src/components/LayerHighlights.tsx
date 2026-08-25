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

import { useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { motion, useInView, useReducedMotion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import LoopVideo from "@/components/LoopVideo";
import { docHref } from "@/lib/docs";
import type { ProjectSite } from "@/lib/docs";

type LayerSlug = ProjectSite;
type NestedMediaLayerSlug = Exclude<LayerSlug, "physics">;
type MediaSlot =
  | "lead"
  | "upper"
  | "middle"
  | "lowerLeft"
  | "lowerRight"
  | "lowerWide"
  | "full";

type TimedOverlayLabel = {
  label: string;
  startsAtSeconds: number;
};

type MediaItem =
  | {
      kind: "video";
      slot: MediaSlot;
      src: string;
      poster: string;
      ariaLabel: string;
      fit?: "cover" | "contain";
      zoom?: number;
      overlayLabel?: string;
      timedOverlayLabels?: readonly TimedOverlayLabel[];
      caption?: string;
    }
  | {
      kind: "image";
      slot: MediaSlot;
      src: string;
      alt: string;
      fit?: "cover" | "contain";
      zoom?: number;
      overlayLabel?: string;
      caption?: string;
    };

type MediaItemForSlot<Slot extends MediaSlot> = MediaItem & { slot: Slot };

type PhysicsMedia = readonly [
  MediaItemForSlot<"lead">,
  MediaItemForSlot<"upper">,
  MediaItemForSlot<"middle">,
  MediaItemForSlot<"lowerLeft">,
  MediaItemForSlot<"lowerRight">,
];

type LayerHighlightBase = {
  id: LayerSlug;
  eyebrow: string;
  heading: string;
  statusLabel?: string;
  intro: ReactNode;
  aspectRatio?: string;
  textFirst?: boolean;
  features: readonly {
    title: string;
    description: ReactNode;
  }[];
  cta: string;
};

type LayerHighlight = LayerHighlightBase &
  (
    | {
        media: PhysicsMedia;
        mediaLayout: "physics";
      }
    | {
        media: readonly MediaItem[];
        mediaLayout?: "bento";
      }
  );

const rootMedia = (file: string) => `${import.meta.env.BASE_URL}img/${file}`;
const layerMedia = (layer: NestedMediaLayerSlug, file: string) =>
  `${import.meta.env.BASE_URL}img/layers/${layer}/${file}`;

function LayerDocLink({
  site,
  children,
}: {
  site: ProjectSite;
  children: ReactNode;
}) {
  return (
    <a
      href={docHref(site)}
      target="_blank"
      rel="noopener noreferrer"
      className="text-foreground decoration-border hover:decoration-foreground underline underline-offset-2 transition-colors"
    >
      {children}
    </a>
  );
}

const LAYER_HIGHLIGHTS: readonly LayerHighlight[] = [
  {
    id: "physics",
    eyebrow: "The engine driving SuperDex",
    heading: "Powered by SuperDex Physics.",
    intro:
      "A contact-first physics engine purpose-built for tactile manipulation, and applicable wherever stable contact and accurate sensing matter. This is the simulation backbone of Project SuperDex.",
    mediaLayout: "physics",
    media: [
      {
        kind: "video",
        slot: "lead",
        src: rootMedia("soft_rigid.mp4"),
        poster: rootMedia("soft_rigid.webp"),
        ariaLabel:
          "Looping demo: soft and rigid body interaction in the SuperDex Physics engine",
      },
      {
        kind: "video",
        slot: "upper",
        src: rootMedia("ropebraid_low.mp4"),
        poster: rootMedia("ropebraid_low.webp"),
        ariaLabel:
          "Looping demo: hands braiding flexible ropes simulated as rod actors",
      },
      {
        kind: "video",
        slot: "middle",
        src: rootMedia("dexterous_multi_actor_manipulation.mp4"),
        poster: rootMedia("dexterous_multi_actor_manipulation.webp"),
        ariaLabel:
          "Looping demo: simulated hands manipulating deformable sheets and rigid objects across multiple materials",
      },
      {
        kind: "video",
        slot: "lowerLeft",
        src: rootMedia("sponge_low.mp4"),
        poster: rootMedia("sponge_low.webp"),
        ariaLabel:
          "Looping demo: a sponge modeled as a soft actor interacting with rigid objects and robotic hands",
      },
      {
        kind: "video",
        slot: "lowerRight",
        src: rootMedia("roperoll_low.mp4"),
        poster: rootMedia("roperoll_low.webp"),
        ariaLabel:
          "Looping demo: hands twisting flexible ropes simulated as rod actors",
      },
    ],
    features: [
      {
        title: "Multi-Physics Simulation",
        description:
          "Unified solver for rigid bodies, soft bodies, rods & tendons, shells & cloth — with more physics on the way.",
      },
      {
        title: "Arbitrary Rigid & Soft Articulations",
        description:
          "Articulated bodies with varying joint types, supporting both rigid links and deformable elements in a single model.",
      },
      {
        title: "Non-Convex Collision",
        description:
          "Accurate contact force distributions for arbitrary geometries including non-convex and deforming objects.",
      },
      {
        title: "Numerical Stability",
        description:
          "Robust simulation without the restrictive time-step stability limits of explicit or semi-implicit methods.",
      },
      {
        title: "Inverse Kinematics",
        description:
          "Constraint-aware inverse-kinematics, built on the same nonlinear optimization core as the forward dynamics, solve physically accurate poses under collision, end-effector, and trajectory constraints.",
      },
    ],
    cta: "Get started with SuperDex Physics",
  },
  {
    id: "robotics",
    eyebrow: "The robotics layer of SuperDex",
    heading: "SuperDex Robotics.",
    intro: (
      <>
        SuperDex Robotics extends the{" "}
        <LayerDocLink site="physics">SuperDex Physics</LayerDocLink>{" "}
        engine to robotics: declarative robot definitions and composition,
        controllers, sensors, and more. This is the robotics toolkit of Project
        SuperDex.
      </>
    ),
    media: [
      {
        kind: "video",
        slot: "lead",
        src: layerMedia("robotics", "robotics_bot_library.mp4"),
        poster: layerMedia("robotics", "robotics_bot_library.webp"),
        ariaLabel:
          "Looping demo: browsing a library of ready-to-load robot arms and dexterous hands in SuperDex Robotics",
        zoom: 1.2,
      },
      {
        kind: "video",
        slot: "upper",
        src: layerMedia("robotics", "bimanual_cube_manipulation.mp4"),
        poster: layerMedia("robotics", "bimanual_cube_manipulation.webp"),
        ariaLabel:
          "Looping demo: two robot arms coordinating to manipulate a cube",
      },
      {
        kind: "image",
        slot: "middle",
        src: layerMedia("robotics", "robot_hand_debug_view.webp"),
        alt: "A dexterous robotic hand shown with link, joint, and collision debug overlays",
      },
      {
        kind: "video",
        slot: "lowerWide",
        src: layerMedia("robotics", "main5.mp4"),
        poster: layerMedia("robotics", "main5.webp"),
        ariaLabel:
          "Looping demo: a robotic hand manipulating a deformable network cable",
      },
    ],
    features: [
      {
        title: "Declarative Bots",
        description: (
          <>
            A human-readable <code>.superdex_bot</code> file describes a complete
            robot — links, joints, and mesh references. It extends the physics
            engine&apos;s articulations with fields robotics workflows demand, and
            is the source of truth from Studio through simulation.
          </>
        ),
      },
      {
        title: "Ready-to-Load Bots",
        description: (
          <>
            A library of bots – arms, robot hands, human hands, sensors, torsos,
            etc. – is readily available, each as a <code>.superdex_bot</code> file
            you can load directly into your simulation scene.
          </>
        ),
      },
      {
        title: "BYO Bots",
        description: (
          <>
            Bring your own robots in from the wider ecosystem. Load a URDF file
            directly at runtime, or upgrade it into a native bot with{" "}
            <LayerDocLink site="studio">SuperDex Studio</LayerDocLink> for
            production-quality collision meshes.
          </>
        ),
      },
      {
        title: "Controllers, Sensors, Actuators",
        description:
          "One uniform, extensible framework for controllers, sensors, and actuators — use the built-ins or register your own. Ships operational-space and joint-space PD controllers to drive robots.",
      },
    ],
    cta: "Get started with SuperDex Robotics",
    aspectRatio: "1 / 1",
    textFirst: true,
  },
  {
    id: "studio",
    eyebrow: "The authoring layer of SuperDex",
    heading: "SuperDex Studio.",
    intro:
      "SuperDex Studio is the content authoring tool for SuperDex. Robots, task objects, and scenes are authored here, to production and simulation-ready quality.",
    media: [
      {
        kind: "video",
        slot: "lead",
        src: layerMedia("studio", "Import2Mesh.mp4"),
        poster: layerMedia("studio", "Import2Mesh.webp"),
        ariaLabel:
          "Importing a robot into SuperDex Studio and meshing its geometry.",
      },
      {
        kind: "image",
        slot: "upper",
        src: layerMedia("studio", "cad_exporter_square.png"),
        alt: "The SuperDex Robot Configuration Exporter configuring link properties on an OpenArm end effector, with its visual and collision meshes overlaid.",
      },
      {
        kind: "image",
        slot: "middle",
        src: layerMedia("studio", "simulation_debug_view.webp"),
        alt: "SuperDex Studio displaying wireframe debug geometry for a scene of rigid and deformable simulation assets.",
        zoom: 1.15,
      },
      {
        kind: "image",
        slot: "lowerLeft",
        src: layerMedia("studio", "bot_link_details.webp"),
        alt: "A Franka arm open in the Bot Editor, with the Bot Hierarchy and Bot Link Details panels beside it.",
      },
      {
        kind: "video",
        slot: "lowerRight",
        src: layerMedia("studio", "force_drag.mp4"),
        poster: layerMedia("studio", "force_drag.webp"),
        ariaLabel: "Force-dragging a robot arm during a live simulation.",
      },
    ],
    features: [
      {
        title: "Compose, Edit, and Combine Bots",
        description: (
          <>
            Assemble complex robots from vetted components or import one from{" "}
            <code>.urdf</code>. Iterate on kinematics and dynamics, joint limits,
            and self-collision until it behaves like the hardware. Bolt a hand
            onto an arm, or replace OEM fingertips with custom sensors using the
            bot mod pipeline.
          </>
        ),
      },
      {
        title: "Create Task Prefabs and Scenes",
        description:
          "Compose rigid, soft, and articulated actors into reusable task prefabs, then assemble those prefabs into scenes for robotics and dexterous manipulation research.",
      },
      {
        title: "Simulate",
        description:
          "Every editor is a live SuperDex Physics scene. Simulate the instant an asset opens, then inspect and debug contact, inertial, and joint behavior on the fly.",
      },
      {
        title: "Edit Meshes",
        description:
          "Turn visual assets and even raw CAD into simulation geometry. Rapidly configurable modifier stacks refine, wrap, re-mesh, and bake signed-distance fields into simulation assets.",
      },
      {
        title: "SuperDex CAD Exporter",
        description:
          "Export assets and robots ready to import into SuperDex Studio directly from our SolidWorks and NX plugins, along with simulation-ready direct-from-CAD meshing.",
      },
    ],
    cta: "Get started with SuperDex Studio",
  },
  {
    id: "lab",
    eyebrow: "The research layer of SuperDex",
    heading: "SuperDex Lab.",
    statusLabel: "Early Preview",
    intro:
      "SuperDex Lab connects simulation and policy development through a Gymnasium-style API for reinforcement learning. It is currently in early preview and will receive substantial improvements. A general abstraction for partially observable Markov decision processes will underpin applications in reinforcement learning, system identification, and model predictive control.",
    media: [
      {
        kind: "video",
        slot: "full",
        src: layerMedia("lab", "sim_to_real_shape_sorting.mp4"),
        poster: layerMedia("lab", "sim_to_real_shape_sorting.webp"),
        ariaLabel:
          "Zero-shot Sim2Real transfer of a shape-sorting policy trained in simulation to a real-world robotic hand. The video shows 8 simulated checkpoints during training, followed by the real-world lab deployment.",
        overlayLabel: "Zero-Shot Sim2Real",
        timedOverlayLabels: [
          { label: "Sim · Checkpoint 1", startsAtSeconds: 0 },
          { label: "Sim · Checkpoint 2", startsAtSeconds: 1.710042 },
          { label: "Sim · Checkpoint 3", startsAtSeconds: 4.004 },
          { label: "Sim · Checkpoint 4", startsAtSeconds: 5.714042 },
          { label: "Sim · Checkpoint 5", startsAtSeconds: 7.632625 },
          { label: "Sim · Checkpoint 6", startsAtSeconds: 9.926583 },
          { label: "Sim · Checkpoint 7", startsAtSeconds: 11.678333 },
          { label: "Sim · Checkpoint 8", startsAtSeconds: 13.179833 },
          { label: "Real", startsAtSeconds: 18.935583 },
        ],
        caption:
          "A shape-sorting manipulation policy trained in simulation with SuperDex Gym and deployed directly on a real-world robotic hand.",
      },
    ],
    features: [
      {
        title: "SuperDex Gym",
        description:
          "A Gymnasium-compatible reinforcement-learning framework built on the SuperDex Physics engine. Train agent controllers in high-fidelity simulation with a suite of manipulation and locomotion environments.",
      },
      {
        title: "RLlib Training",
        description:
          "An off-the-shelf Ray/RLlib integration for RL training straight from the Gym environments.",
      },
    ],
    cta: "Get started with SuperDex Lab",
    aspectRatio: "16 / 9",
    textFirst: true,
  },
];

const MEDIA_SLOT_CLASSES: Record<MediaSlot, string> = {
  lead:
    "col-span-2 aspect-video sm:col-start-1 sm:col-end-5 sm:row-start-1 sm:row-end-3 sm:aspect-auto",
  upper:
    "aspect-square sm:col-start-5 sm:col-end-7 sm:row-start-1 sm:row-end-2 sm:aspect-auto",
  middle:
    "aspect-square sm:col-start-5 sm:col-end-7 sm:row-start-2 sm:row-end-3 sm:aspect-auto",
  lowerLeft:
    "aspect-square sm:col-start-1 sm:col-end-4 sm:row-start-3 sm:row-end-4 sm:aspect-auto",
  lowerRight:
    "aspect-square sm:col-start-4 sm:col-end-7 sm:row-start-3 sm:row-end-4 sm:aspect-auto",
  lowerWide:
    "col-span-2 aspect-video sm:col-start-1 sm:col-end-7 sm:row-start-3 sm:row-end-4 sm:aspect-auto",
  full:
    "col-span-2 aspect-video sm:col-start-1 sm:col-end-7 sm:row-start-1 sm:row-end-4 sm:aspect-auto",
};

function MediaFigure({
  item,
  reducedMotion,
  className = "",
  mediaClassName = "block h-full w-full",
  captionAspectRatio,
}: {
  item: MediaItem;
  reducedMotion: boolean;
  className?: string;
  mediaClassName?: string;
  captionAspectRatio?: string;
}) {
  const objectFitClass = item.fit === "contain" ? "object-contain" : "object-cover";
  const style = item.zoom ? { transform: `scale(${item.zoom})` } : undefined;
  const timedOverlayLabels =
    item.kind === "video" ? item.timedOverlayLabels : undefined;
  const [activeTimedOverlayIndex, setActiveTimedOverlayIndex] = useState(0);
  const activeTimedOverlayLabel =
    timedOverlayLabels?.[activeTimedOverlayIndex]?.label;
  const media =
    item.kind === "video" ? (
      <LoopVideo
        src={item.src}
        poster={item.poster}
        ariaLabel={item.ariaLabel}
        reducedMotion={reducedMotion}
        className={`${mediaClassName} ${objectFitClass}`}
        style={style}
        onPlaybackTime={
          timedOverlayLabels
            ? (currentTime) => {
                let nextIndex = 0;
                for (let index = 1; index < timedOverlayLabels.length; index++) {
                  if (currentTime < timedOverlayLabels[index].startsAtSeconds) {
                    break;
                  }
                  nextIndex = index;
                }
                setActiveTimedOverlayIndex((currentIndex) =>
                  currentIndex === nextIndex ? currentIndex : nextIndex,
                );
              }
            : undefined
        }
      />
    ) : (
      <img
        src={item.src}
        alt={item.alt}
        loading="lazy"
        decoding="async"
        className={`${mediaClassName} ${objectFitClass}`}
        style={style}
      />
    );
  const overlay = item.overlayLabel ? (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute left-3 top-3 z-10 inline-flex items-center gap-2 rounded-full bg-black/70 px-3 py-1.5 text-[0.68rem] font-semibold tracking-[0.12em] text-white backdrop-blur-sm sm:left-4 sm:top-4 sm:gap-2.5 sm:px-4 sm:py-2 sm:text-xs"
    >
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 rounded-full bg-emerald-400 sm:h-2 sm:w-2"
      />
      {item.overlayLabel}
    </div>
  ) : null;
  const timedOverlay = activeTimedOverlayLabel ? (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute bottom-3 right-3 z-10 rounded-full bg-black/70 px-3 py-1.5 text-[0.68rem] font-semibold tracking-[0.12em] text-white backdrop-blur-sm sm:bottom-4 sm:right-4 sm:px-4 sm:py-2 sm:text-xs"
    >
      {activeTimedOverlayLabel}
    </div>
  ) : null;

  if (item.caption) {
    return (
      <figure className={`m-0 flex min-h-0 flex-col gap-3 ${className}`}>
        <div
          className={`screenshot-frame relative min-h-0 flex-1 ${
            item.fit === "contain" ? "bg-[oklch(0.12_0_0)]" : ""
          }`}
          style={{ aspectRatio: captionAspectRatio }}
        >
          {media}
          {overlay}
          {timedOverlay}
        </div>
        <figcaption className="body-text px-1 text-center text-sm italic">
          {item.caption}
        </figcaption>
      </figure>
    );
  }

  return (
    <figure
      className={`screenshot-frame relative m-0 min-h-0 ${className} ${
        item.fit === "contain"
          ? "bg-[oklch(0.12_0_0)]"
          : item.kind === "video"
            ? "bg-muted"
            : ""
      }`}
    >
      {media}
      {overlay}
      {timedOverlay}
    </figure>
  );
}

function PhysicsLayerMedia({
  media,
  reducedMotion,
}: {
  media: PhysicsMedia;
  reducedMotion: boolean;
}) {
  const [lead, upper, middle, lowerLeft, lowerRight] = media;

  return (
    <div className="lg:grid lg:aspect-square lg:grid-rows-[2fr_1fr] lg:gap-3">
      <div className="mb-3 grid grid-cols-1 gap-3 lg:mb-0 lg:min-h-0 lg:grid-cols-3">
        <div className="lg:col-span-2 lg:min-h-0">
          <MediaFigure
            item={lead}
            reducedMotion={reducedMotion}
            className="flex h-full flex-col"
            mediaClassName="h-full w-full flex-1"
          />
        </div>
        <div className="flex min-h-0 flex-col gap-3">
          <MediaFigure
            item={upper}
            reducedMotion={reducedMotion}
            className="flex min-h-0 flex-1 flex-col"
            mediaClassName="h-full w-full flex-1"
          />
          <MediaFigure
            item={middle}
            reducedMotion={reducedMotion}
            className="flex min-h-0 flex-1 flex-col"
            mediaClassName="h-full w-full flex-1"
          />
        </div>
      </div>
      <div className="grid min-h-0 grid-cols-2 gap-3">
        <MediaFigure
          item={lowerLeft}
          reducedMotion={reducedMotion}
          mediaClassName="h-full w-full"
        />
        <MediaFigure
          item={lowerRight}
          reducedMotion={reducedMotion}
          mediaClassName="h-full w-full"
        />
      </div>
    </div>
  );
}

function LayerMedia({
  layer,
  reducedMotion,
}: {
  layer: LayerHighlight;
  reducedMotion: boolean;
}) {
  if (layer.mediaLayout === "physics") {
    return <PhysicsLayerMedia media={layer.media} reducedMotion={reducedMotion} />;
  }

  if (layer.media.length === 1 && layer.media[0].slot === "full") {
    const item = layer.media[0];
    return (
      <MediaFigure
        item={item}
        reducedMotion={reducedMotion}
        className={item.caption ? "" : "aspect-video"}
        captionAspectRatio={
          item.caption ? layer.aspectRatio ?? "16 / 9" : undefined
        }
      />
    );
  }

  return (
    <div
      className="grid grid-cols-2 gap-2.5 sm:grid-cols-6 sm:grid-rows-[1fr_1fr_1.35fr] sm:[aspect-ratio:var(--layer-media-aspect)]"
      style={{
        "--layer-media-aspect": layer.aspectRatio ?? "1 / 1.15",
      } as CSSProperties}
    >
      {layer.media.map((item) => (
        <MediaFigure
          key={item.src}
          item={item}
          reducedMotion={reducedMotion}
          className={MEDIA_SLOT_CLASSES[item.slot]}
        />
      ))}
    </div>
  );
}

function LayerHighlightSection({
  layer,
  reducedMotion,
}: {
  layer: LayerHighlight;
  reducedMotion: boolean;
}) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const animationDuration = reducedMotion ? 0 : 0.6;
  const textFirst = layer.textFirst ?? false;

  return (
    <section
      id={layer.id}
      aria-labelledby={`${layer.id}-heading`}
      className="bg-background py-12 lg:py-16"
    >
      <div className="container" ref={ref}>
        <motion.div
          initial={reducedMotion ? false : { opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: reducedMotion ? 0 : 0.55 }}
          className="mb-14"
        >
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="flex flex-col gap-4">
              <p className="section-eyebrow">{layer.eyebrow}</p>
              <div className="flex flex-wrap items-center gap-3">
                <h2
                  id={`${layer.id}-heading`}
                  className="display-heading"
                  style={{ fontSize: "clamp(2rem, 3.5vw, 3rem)" }}
                >
                  {layer.heading}
                </h2>
                {layer.statusLabel && (
                  <span className="tag preview-surface shrink-0">
                    {layer.statusLabel}
                  </span>
                )}
              </div>
            </div>
            <p className="body-text max-w-md">{layer.intro}</p>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 items-start gap-12 lg:grid-cols-2 lg:gap-20">
          <motion.div
            initial={
              reducedMotion ? false : { opacity: 0, x: textFirst ? 20 : -20 }
            }
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: animationDuration }}
            className={textFirst ? "lg:order-2" : "lg:order-1"}
          >
            <LayerMedia layer={layer} reducedMotion={reducedMotion} />
          </motion.div>

          <motion.div
            initial={
              reducedMotion ? false : { opacity: 0, x: textFirst ? -20 : 20 }
            }
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{
              duration: animationDuration,
              delay: reducedMotion ? 0 : 0.1,
            }}
            className={`flex flex-col ${
              textFirst ? "lg:order-1" : "lg:order-2"
            }`}
          >
            <div className="divide-border divide-y">
              {layer.features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={reducedMotion ? false : { opacity: 0, y: 16 }}
                  animate={inView ? { opacity: 1, y: 0 } : {}}
                  transition={{
                    duration: reducedMotion ? 0 : 0.45,
                    delay: reducedMotion ? 0 : 0.15 + index * 0.08,
                  }}
                  className="py-6"
                >
                  <h3
                    className="text-card-foreground mb-1.5 font-semibold"
                    style={{
                      fontFamily: "'DM Sans', sans-serif",
                      fontSize: "0.95rem",
                    }}
                  >
                    {feature.title}
                  </h3>
                  <p className="body-text" style={{ fontSize: "0.88rem" }}>
                    {feature.description}
                  </p>
                </motion.div>
              ))}
            </div>

            <div className="pt-8">
              <a
                href={docHref(layer.id)}
                target="_blank"
                rel="noopener noreferrer"
                className="link-arrow no-underline"
              >
                {layer.cta} <ArrowRight aria-hidden="true" size={12} />
              </a>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

export default function LayerHighlights() {
  const reducedMotion = useReducedMotion() ?? false;

  return (
    <>
      {LAYER_HIGHLIGHTS.map((layer) => (
        <LayerHighlightSection
          key={layer.id}
          layer={layer}
          reducedMotion={reducedMotion}
        />
      ))}
    </>
  );
}
