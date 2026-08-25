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
 * SIMULATION GALLERY — Full-width carousel with side-by-side video + description
 */
import {
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import {
  motion,
  useInView,
  useReducedMotion,
  AnimatePresence,
} from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Link } from "wouter";
import LoopVideo from "@/components/LoopVideo";

const DEXTEROUS_MULTI_ACTOR_MANIPULATION_VIDEO = `${import.meta.env.BASE_URL}img/gallery/dexterous_multi_actor_manipulation.mp4`;
const CONTACT_VISUALIZATION_VIDEO = `${import.meta.env.BASE_URL}img/gallery/contact_visualization.mp4`;
const THREADED_ROD_AND_NUT_VIDEO = `${import.meta.env.BASE_URL}img/gallery/threadedrodandnut_low.mp4`;
const SPONGE_VIDEO = `${import.meta.env.BASE_URL}img/gallery/sponge_low.mp4`;
const DEFORMABLE_FINGERTIPS_RIGID_GEOMETRY_VIDEO = `${import.meta.env.BASE_URL}img/gallery/deformable_fingertips_rigid_geometry.mp4`;
const NETWORKCABLE_VIDEO = `${import.meta.env.BASE_URL}img/gallery/networkcable_low.mp4`;
const FRUITBAG_VIDEO = `${import.meta.env.BASE_URL}img/gallery/fruitbag_low.mp4`;
const CEREALBOX_VIDEO = `${import.meta.env.BASE_URL}img/gallery/cerealbox_low.mp4`;

type Slide = {
  src: string;
  poster: string;
  label: string;
  description: ReactNode;
};

const slides: readonly Slide[] = [
  {
    src: DEXTEROUS_MULTI_ACTOR_MANIPULATION_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/dexterous_multi_actor_manipulation.webp`,
    label: "Fine-grained dexterous manipulation",
    description:
      "Fine-grained hand–object interactions support dexterous manipulation of both deformable and rigid objects, from folding a napkin to placing shapes into a shape sorter.",
  },
  {
    src: CONTACT_VISUALIZATION_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/contact_visualization.webp`,
    label: "Distributed contact forces",
    description:
      "Contact forces are resolved as a spatially varying traction field (normal and friction components) over the surface, capturing the distribution of loads across the contact region.",
  },
  {
    src: THREADED_ROD_AND_NUT_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/threadedrodandnut_low.webp`,
    label: "Threading a nut",
    description:
      "Stable, accurate contact simulation for fine-grained, complex, non-convex geometries such as a nut threading onto a bolt.",
  },
  {
    src: SPONGE_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/sponge_low.webp`,
    label: "Sponge modeled as a soft actor",
    description:
      "Soft actors can simulate three-dimensional deformable objects, such as a sponge that compresses and changes shape during manipulation.",
  },
  {
    src: DEFORMABLE_FINGERTIPS_RIGID_GEOMETRY_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/deformable_fingertips_rigid_geometry.webp`,
    label: "Soft robotic fingertips",
    description:
      "Soft articulations can model compliant components in end effectors and tactile sensors, such as robotic fingertips that deform around rigid edges during pressing and sliding.",
  },
  {
    src: NETWORKCABLE_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/networkcable_low.webp`,
    label: "Cable modeled as a rod actor",
    description:
      "Rod actors can simulate cables, tendons, springs, ropes, and other slender deformable structures.",
  },
  {
    src: FRUITBAG_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/fruitbag_low.webp`,
    label: "Fabric bag modeled as a shell actor",
    description:
      "Shell actors can simulate fabric, membranes, and other structures with low bending stiffness, such as fabric bags.",
  },
  {
    src: CEREALBOX_VIDEO,
    poster: `${import.meta.env.BASE_URL}img/gallery/posters/cerealbox_low.webp`,
    label: "Cereal box modeled as a shell actor",
    description:
      "Shell actors can also simulate deformable plates and thin-walled structures with higher bending stiffness, such as cardboard boxes.",
  },
];

export default function SimulationShowcase() {
  const reducedMotion = useReducedMotion() ?? false;
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [activeSlide, setActiveSlide] = useState(0);

  // Preserve per-slide video playback position across mount/unmount.
  const videoTimesRef = useRef<Map<number, number>>(new Map());

  const prev = () =>
    setActiveSlide((s) => (s - 1 + slides.length) % slides.length);
  const next = () => setActiveSlide((s) => (s + 1) % slides.length);

  const onCarouselKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      prev();
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      next();
    }
  };

  const slide = slides[activeSlide];

  return (
    <section
      id="showcase"
      className="py-12 lg:py-16"
      style={{ background: "oklch(1 0 0)" }}
    >
      <div className="container" ref={ref}>
        {/* Header */}
        <motion.div
          initial={reducedMotion ? false : { opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: reducedMotion ? 0 : 0.55 }}
          className="mb-14"
        >
          <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
            <div className="flex flex-col gap-4">
              <p className="section-eyebrow">Gallery</p>
              <h2
                className="display-heading"
                style={{ fontSize: "clamp(2rem, 3.5vw, 3rem)" }}
              >
                See it in action!
              </h2>
            </div>
            <p className="body-text max-w-md">
              This gallery shows interactive virtual teleoperation powered by
              SuperDex Physics, with a person using a Meta Quest 3 headset to
              perform a dexterous-manipulation task. The physics is{" "}
              <strong className="font-semibold">simulated in real time</strong>{" "}
              for synthetic data collection and policy development.
            </p>
          </div>
        </motion.div>

        {/* Full-width carousel */}
        <motion.div
          initial={reducedMotion ? false : { opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: reducedMotion ? 0 : 0.6 }}
          role="region"
          aria-roledescription="carousel"
          aria-label="Gallery"
          tabIndex={0}
          onKeyDown={onCarouselKeyDown}
          className="focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[oklch(0.28_0_0)] rounded-xl"
        >
          <p className="sr-only" aria-live="polite" aria-atomic="true">
            Slide {activeSlide + 1} of {slides.length}: {slide.label}
          </p>
          {/* Video + description side by side */}
          <div
            className="rounded-xl border overflow-hidden"
            style={{
              borderColor: "oklch(0.88 0 0)",
              background: "oklch(0.99 0 0)",
            }}
          >
            <div className="grid grid-cols-1 lg:grid-cols-5">
              {/* Video — 3/5 width */}
              <div className="lg:col-span-3 relative aspect-video bg-[oklch(0.12_0_0)] overflow-hidden">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeSlide}
                    initial={reducedMotion ? false : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={reducedMotion ? { opacity: 1 } : { opacity: 0 }}
                    transition={{ duration: reducedMotion ? 0 : 0.3 }}
                    className="w-full h-full"
                  >
                    <LoopVideo
                      src={slide.src}
                      poster={slide.poster}
                      ariaLabel={`Real-time simulation: ${slide.label}`}
                      reducedMotion={reducedMotion}
                      className="w-full h-full object-cover"
                      onLoadedMetadata={(e) => {
                        const saved = videoTimesRef.current.get(activeSlide);
                        if (saved != null) {
                          e.currentTarget.currentTime = saved;
                        }
                      }}
                      onTimeUpdate={(e) => {
                        videoTimesRef.current.set(
                          activeSlide,
                          e.currentTarget.currentTime,
                        );
                      }}
                    />
                  </motion.div>
                </AnimatePresence>
                <div className="pointer-events-none absolute left-3 top-3 z-10 inline-flex items-center gap-2 rounded-full bg-black/70 px-3 py-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.12em] text-white backdrop-blur-sm sm:left-4 sm:top-4 sm:gap-2.5 sm:px-4 sm:py-2 sm:text-xs">
                  <span
                    aria-hidden="true"
                    className="h-1.5 w-1.5 rounded-full bg-emerald-400 sm:h-2 sm:w-2"
                  />
                  Real-time simulation
                </div>
              </div>

              {/* Description panel — 2/5 width */}
              <div className="lg:col-span-2 flex flex-col justify-between p-6 lg:p-8">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeSlide}
                    initial={
                      reducedMotion ? false : { opacity: 0, y: 10 }
                    }
                    animate={{ opacity: 1, y: 0 }}
                    exit={
                      reducedMotion
                        ? { opacity: 1, y: 0 }
                        : { opacity: 0, y: -10 }
                    }
                    transition={{ duration: reducedMotion ? 0 : 0.25 }}
                  >
                    <p
                      className="section-eyebrow mb-3"
                      style={{ fontSize: "0.7rem" }}
                    >
                      {String(activeSlide + 1).padStart(2, "0")} / {String(slides.length).padStart(2, "0")}
                    </p>
                    <h3
                      className="font-semibold text-lg mb-3"
                      style={{
                        fontFamily: "'Instrument Serif', serif",
                        color: "oklch(0.18 0 0)",
                        fontSize: "clamp(1.25rem, 2vw, 1.5rem)",
                      }}
                    >
                      {slide.label}
                    </h3>
                    <p
                      className="text-sm leading-relaxed"
                      style={{
                        fontFamily: "'DM Sans', sans-serif",
                        color: "oklch(0.45 0 0)",
                      }}
                    >
                      {slide.description}
                    </p>
                  </motion.div>
                </AnimatePresence>

                {/* Navigation controls */}
                <div className="flex items-center gap-2 mt-6">
                  <button
                    onClick={prev}
                    aria-label="Previous slide"
                    className="w-9 h-9 rounded border flex items-center justify-center transition-colors"
                    style={{
                      borderColor: "oklch(0.84 0 0)",
                      color: "oklch(0.45 0 0)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor =
                        "oklch(0.65 0 0)";
                      e.currentTarget.style.color = "oklch(0.18 0 0)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor =
                        "oklch(0.84 0 0)";
                      e.currentTarget.style.color = "oklch(0.45 0 0)";
                    }}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <button
                    onClick={next}
                    aria-label="Next slide"
                    className="w-9 h-9 rounded border flex items-center justify-center transition-colors"
                    style={{
                      borderColor: "oklch(0.84 0 0)",
                      color: "oklch(0.45 0 0)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor =
                        "oklch(0.65 0 0)";
                      e.currentTarget.style.color = "oklch(0.18 0 0)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor =
                        "oklch(0.84 0 0)";
                      e.currentTarget.style.color = "oklch(0.45 0 0)";
                    }}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Thumbnail strip */}
          <div className="flex gap-2 mt-4 overflow-x-auto pb-2 lg:overflow-visible lg:pb-0">
            {slides.map((s, i) => (
              <button
                key={i}
                onClick={() => setActiveSlide(i)}
                aria-label={`Show real-time simulation: ${s.label}`}
                aria-current={i === activeSlide ? "true" : undefined}
                className="w-32 min-w-32 shrink-0 aspect-video rounded overflow-hidden border transition-all duration-200 lg:w-auto lg:min-w-0 lg:flex-1"
                style={{
                  borderColor:
                    i === activeSlide
                      ? "oklch(0.28 0 0)"
                      : "oklch(0.84 0 0)",
                  opacity: i === activeSlide ? 1 : 0.55,
                }}
              >
                <img
                  src={s.poster}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                  decoding="async"
                  className="w-full h-full object-cover"
                />
              </button>
            ))}
          </div>
        </motion.div>

        <p
          className="mt-6 text-xs leading-relaxed"
          style={{
            fontFamily: "'DM Sans', sans-serif",
            color: "oklch(0.45 0 0)",
          }}
        >
          Project SuperDex webpages and documentation may include open-source or third-party 3D models and other visual assets.{" "}
          <Link
            href="/credits"
            className="font-medium underline underline-offset-4 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
            style={{ color: "oklch(0.28 0 0)" }}
          >
            See credits.
          </Link>
        </p>
      </div>
    </section>
  );
}
