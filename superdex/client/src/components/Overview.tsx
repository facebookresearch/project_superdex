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
 * PLATFORM OVERVIEW — Three pillars: Simulate, Learn, Deploy
 * Clean card layout on warm off-white background
 */
import { useRef } from "react";
import { Link } from "wouter";
import { motion, useInView } from "framer-motion";
import { Cpu, Video, Brain, ArrowRight } from "lucide-react";

const pillars = [
  {
    icon: Cpu,
    number: "01",
    title: "Simulate",
    subtitle: "State-of-the-art physics",
    description:
      "SuperDex Physics delivers stable, accurate contact physics for the most demanding dexterous interactions — multi-finger grasps, in-hand reorientation, and non-convex contact.",
    tags: [ "Soft/Rigid Articulations", "Soft Bodies", "Rods/tendons", "Shells/cloth", "Tactile Sensors", "Soft contact", "Non-convex Collision"],
  },
  {
    icon: Video,
    number: "02",
    title: "Teleoperate",
    subtitle: "Synthetic data at scale",
    description:
      "Virtual teleoperation lets human operators drive simulated robots, generating demonstration datasets at a fraction of the cost and risk of real-world collection.",
    tags: ["VR Teleoperation", "Haptic Feedback", "Demonstration Data", "Scalable Collection"],
  },
  {
    icon: Brain,
    number: "03",
    title: "Train",
    subtitle: "Reinforcement learning at scale",
    description:
      "Run batched simulations with Gymnasium-compatible APIs and Ray/RLlib integration. Train manipulation policies across vectorized environments with scene sharing and multi-threaded execution.",
    tags: ["Batched Simulation", "Gymnasium API", "Ray / RLlib", "Vectorized Training"],
  },
];

export default function Overview() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section
      id="overview"
      className="py-12 lg:py-16"
      style={{ background: "oklch(1 0 0)" }}
    >
      <div className="container" ref={ref}>
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.55 }}
          className="mb-16"
        >
          <p className="section-eyebrow mb-4">Overview</p>
          <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-6">
            <h2
              className="display-heading"
              style={{ fontSize: "clamp(2rem, 3.5vw, 3rem)" }}
            >
              Welcome to Project SuperDex.
            </h2>
            <div className="max-w-md">
              <p className="body-text">
                SuperDex brings together a purpose-built physics engine,
                robotics authoring tools, and a reinforcement learning interface
                in a unified simulation platform, with VR-based teleoperation
                and additional capabilities planned for future releases.
              </p>
              {/* Routes to the Roadmap tab */}
              <p className="mt-4">
                <Link
                  href="/roadmap"
                  className="link-arrow"
                  style={{ display: "inline-flex" }}
                >
                  Explore the roadmap <ArrowRight size={12} />
                </Link>
              </p>
            </div>
          </div>
        </motion.div>

        {/* Pillars */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-12">
          {pillars.map((pillar, i) => {
            const Icon = pillar.icon;
            return (
              <motion.div
                key={pillar.number}
                initial={{ opacity: 0, y: 30 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.55, delay: i * 0.12 }}
                className="flex flex-col"
              >
                {/* Number + icon */}
                <div className="flex items-center justify-between mb-6">
                  <span
                    className="text-xs font-medium"
                    style={{
                      fontFamily: "'DM Sans', sans-serif",
                      color: "oklch(0.65 0 0)",
                      letterSpacing: "0.1em",
                    }}
                  >
                    {pillar.number}
                  </span>
                  <div
                    className="w-9 h-9 rounded-lg flex items-center justify-center"
                    style={{ background: "oklch(0.93 0 0)" }}
                  >
                    <Icon size={17} style={{ color: "oklch(0.28 0 0)" }} />
                  </div>
                </div>

                {/* Title */}
                <p
                  className="text-xs font-medium mb-1"
                  style={{
                    color: "oklch(0.55 0 0)",
                    fontFamily: "'DM Sans', sans-serif",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                  }}
                >
                  {pillar.subtitle}
                </p>
                <h3
                  className="display-heading mb-3"
                  style={{ fontSize: "1.6rem" }}
                >
                  {pillar.title}
                </h3>
                <p className="body-text text-sm mb-6 flex-1">{pillar.description}</p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2">
                  {pillar.tags.map((tag) => (
                    <span key={tag} className="tag">
                      {tag}
                    </span>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
