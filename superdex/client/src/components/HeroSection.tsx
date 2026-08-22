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
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { useLocation } from "wouter";

const HERO_IMAGE = `${import.meta.env.BASE_URL}img/hero_hand_v21.webp`;

export default function HeroSection() {
  const [, setLocation] = useLocation();

  const handleScrollDown = () => {
    const el = document.querySelector("#overview");
    if (el) el.scrollIntoView({ behavior: "smooth" });
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
              onClick={handleGetStarted}
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
              Get Started
              <ArrowRight size={14} />
            </button>
            <button
              onClick={handleScrollDown}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded text-sm font-medium border transition-all duration-200"
              style={{
                borderColor: "oklch(0.82 0 0)",
                color: "oklch(0.35 0 0)",
                fontFamily: "'DM Sans', sans-serif",
                background: "transparent",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "oklch(0.55 0 0)";
                e.currentTarget.style.color = "oklch(0.15 0 0)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "oklch(0.82 0 0)";
                e.currentTarget.style.color = "oklch(0.35 0 0)";
              }}
            >
              Learn More
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
    </section>
  );
}
