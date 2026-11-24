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
 * HERO SECTION — Clean editorial layout
 * Layout: Left text block + right full-bleed image
 * Background: Warm off-white
 */
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowDown, ArrowRight, ChevronDown } from "lucide-react";
import { useEffect, useState } from "react";
import { useLocation } from "wouter";

const HERO_IMAGE = `${import.meta.env.BASE_URL}img/hero_hand_v21.webp`;

// Matches the Navbar's solid-background threshold.
const SCROLL_CUE_HIDE_OFFSET = 40;

export default function HeroSection() {
  const [, setLocation] = useLocation();
  const reduceMotion = useReducedMotion();
  const [showScrollCue, setShowScrollCue] = useState(true);

  useEffect(() => {
    const onScroll = () =>
      setShowScrollCue(window.scrollY <= SCROLL_CUE_HIDE_OFFSET);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleScrollDown = () => {
    const el = document.querySelector("#overview");
    if (el) el.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  };

  const handleGetStarted = () => {
    setLocation("/get-started");
  };

  return (
    <section
      // The tall desktop row keeps the text vertically centered beside the
      // landscape artwork. Mobile uses a shorter row and hides the artwork.
      className="relative min-h-[85vh] lg:min-h-[110vh] flex flex-col"
      style={{ background: "oklch(1 0 0)" }}
    >
      {/* Main hero content */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2">
        {/* Left: Text */}
        <div className="flex flex-col justify-center px-6 sm:px-12 lg:px-16 xl:px-24 py-16 lg:py-0">
          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="section-eyebrow mb-5"
          >
            Project SuperDex
          </motion.p>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="display-heading mb-6"
            style={{ fontSize: "clamp(2.8rem, 5vw, 4.5rem)" }}
          >
            A unified platform<br />
            for dexterous manipulation research
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.42 }}
            className="body-text mb-8 max-w-md"
            style={{ fontSize: "1.05rem" }}
          >
            SuperDex is an open-source simulation platform from Meta for robotic
            dexterous manipulation, built around a custom physics engine for
            complex, contact-first interactions.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="flex flex-wrap gap-3"
          >
            <button
              onClick={handleScrollDown}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded text-sm font-medium transition-all duration-200"
              style={{
                background: "oklch(0.15 0 0)",
                color: "oklch(0.97 0 0)",
                fontFamily: "'DM Sans', sans-serif",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "oklch(0.25 0 0)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "oklch(0.15 0 0)")
              }
            >
              Learn More
              <ArrowDown aria-hidden="true" size={14} />
            </button>
            <button
              onClick={handleGetStarted}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded text-sm font-medium border border-border bg-background text-foreground hover:border-muted-foreground transition-all duration-200"
              style={{ fontFamily: "'DM Sans', sans-serif" }}
            >
              Get Started
              <ArrowRight aria-hidden="true" size={14} />
            </button>
          </motion.div>
        </div>

        {/* Right: Image */}
        <motion.div
          initial={{ opacity: 0, scale: 1.02 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="relative hidden overflow-hidden lg:block"
        >
          {/* Scale from the right edge so the empty left canvas clips away while
              the hand and fingertips stay in frame. */}
          <img
            src={HERO_IMAGE}
            alt="Line drawing of a dexterous robotic hand reaching down to a surface"
            className="h-full w-full origin-right scale-[1.7] object-contain object-right"
            style={{ minHeight: "100%" }}
          />
        </motion.div>
      </div>

      {/* Scroll cue: the desktop hero is taller than the viewport, so pin the
          cue to the viewport bottom until the visitor starts scrolling. Phones
          already show the next section, so the cue is desktop-only. It is
          mouse-only because "Learn More" does the same thing for keyboard and
          screen-reader users. */}
      <AnimatePresence>
        {showScrollCue && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{
              opacity: 1,
              transition: {
                duration: reduceMotion ? 0 : 0.5,
                delay: reduceMotion ? 0 : 1.1,
              },
            }}
            exit={{
              opacity: 0,
              transition: { duration: reduceMotion ? 0 : 0.2 },
            }}
            className="pointer-events-none fixed inset-x-0 bottom-6 z-30 hidden justify-center lg:flex"
          >
            <button
              type="button"
              onClick={handleScrollDown}
              tabIndex={-1}
              aria-hidden="true"
              className="pointer-events-auto flex flex-col items-center gap-1 rounded px-3 py-1 text-muted-foreground transition-colors hover:text-foreground"
            >
              <span
                className="text-[0.7rem] font-medium uppercase"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  letterSpacing: "0.14em",
                }}
              >
                Scroll
              </span>
              <motion.span
                aria-hidden="true"
                animate={reduceMotion ? undefined : { y: [0, 5, 0] }}
                transition={
                  reduceMotion
                    ? undefined
                    : { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                }
                className="flex"
              >
                <ChevronDown size={18} />
              </motion.span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
