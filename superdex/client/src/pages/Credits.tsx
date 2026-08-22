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

import { useEffect } from "react";
import { ArrowLeft } from "lucide-react";
import { useLocation } from "wouter";
import FooterSection from "@/components/FooterSection";
import Navbar from "@/components/Navbar";
import fontLicenseUrl from "@/fonts/OFL.txt?url";
import { scrollToHashOnHome } from "@/lib/scroll";

const NAV_HEIGHT = 64;

const assetCredits = [
  {
    label: "NIST — Assembly Task Board",
    href: "https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly",
  },
  {
    label: "Seed Robotics — SINGLEX-3 tactile sensor assets (Apache-2.0)",
    href: "https://www.seedrobotics.com/",
  },
  {
    label:
      "Franka Robotics — FR3 and FR3 V2 model assets (Copyright 2023 Franka Robotics GmbH, Apache-2.0)",
    href: "https://github.com/frankarobotics/franka_description/tree/main",
  },
  {
    label:
      "ROS-Industrial — Robotiq 2F-85 Adaptive Gripper model assets (Copyright 2013 ROS-Industrial, BSD-2-Clause)",
    href: "https://github.com/ros-industrial-attic/robotiq",
  },
  {
    label: "Enactic — OpenArm V2.0 model assets (Apache-2.0)",
    href: "https://openarm.dev",
  },
  {
    label:
      "Wuji Technology — Wuji Hand 2 (Beta 1) model assets (Copyright 2025 Wuji Technology, MIT)",
    href: "https://github.com/wuji-technology/wuji-description",
  },
  {
    label: "Komira — RubberDuck 3D model",
    href: "https://sketchfab.com/3d-models/rubberduck-5b20ca71fe4f475e8838420fd66519d2",
  },
] as const;

const creditLinkClass =
  "underline underline-offset-4 transition-colors hover:text-[oklch(0.18_0_0)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2";

export default function CreditsPage() {
  const [, setLocation] = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "oklch(1 0 0)" }}
    >
      <Navbar />
      <div style={{ height: NAV_HEIGHT }} />
      <main className="flex-1 py-16 lg:py-24">
        <div className="container max-w-4xl">
          <button
            type="button"
            onClick={() => scrollToHashOnHome("#showcase", setLocation)}
            className="inline-flex items-center gap-2 mb-12 text-sm font-medium underline underline-offset-4 transition-colors hover:text-[oklch(0.45_0_0)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4"
            style={{
              fontFamily: "'DM Sans', sans-serif",
              color: "oklch(0.28 0 0)",
            }}
          >
            <ArrowLeft size={14} aria-hidden="true" />
            Back to Gallery
          </button>

          <p className="section-eyebrow mb-4">Credits</p>
          <h1
            className="display-heading mb-6"
            style={{ fontSize: "clamp(2.4rem, 5vw, 4.5rem)" }}
          >
            Third-party credits
          </h1>
          <p className="body-text max-w-2xl">
            Project SuperDex webpages and documentation include third-party 3D
            models, visual assets, and fonts credited below.
          </p>

          <ul className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-12">
            <li
              className="md:col-span-2 rounded-xl border p-6 lg:p-8"
              style={{
                borderColor: "oklch(0.88 0 0)",
                background: "oklch(0.99 0 0)",
              }}
            >
              <p
                className="text-sm leading-relaxed"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
              >
                {"The 3D models of the Tesollo hand/gripper are provided by Tesollo, Inc. ("}
                <a
                  href="https://tesollo.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={creditLinkClass}
                >
                  https://tesollo.com
                </a>
                {") and are used with permission. The Tesollo name, logo, and DG-5F-M and DG-5F-S product names are used only for identification and do not imply endorsement, certification, sponsorship, or an official partnership with Tesollo Inc."}
              </p>
            </li>
            <li
              className="md:col-span-2 rounded-xl border p-6 lg:p-8"
              style={{
                borderColor: "oklch(0.88 0 0)",
                background: "oklch(0.99 0 0)",
              }}
            >
              <p
                className="text-sm leading-relaxed"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
              >
                Seed Robotics and SINGLEX-3 names are used solely for accurate
                identification and attribution. Their use does not imply
                endorsement, certification, sponsorship, or an official
                partnership with Seed Robotics.
              </p>
            </li>
            <li
              className="md:col-span-2 rounded-xl border p-6 lg:p-8"
              style={{
                borderColor: "oklch(0.88 0 0)",
                background: "oklch(0.99 0 0)",
              }}
            >
              <p
                className="text-sm leading-relaxed"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
              >
                <a
                  href="https://www.allegrohand.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className={creditLinkClass}
                >
                  Allegro Hand V5
                </a>{" "}
                assets are provided by Wonik Robotics. Copyright © 2026 Wonik
                Robotics. They are licensed for non-commercial research,
                educational, evaluation, and internal-development use; commercial
                use requires Wonik Robotics&apos; prior written permission. Meta
                modified the original assets for SuperDex, including collision
                geometry, kinematic and dynamic parameters, and mesh formats.
              </p>
            </li>
            {assetCredits.map((credit) => (
              <li
                key={credit.href}
                className="rounded-xl border p-6"
                style={{
                  borderColor: "oklch(0.88 0 0)",
                  background: "oklch(0.99 0 0)",
                }}
              >
                <a
                  href={credit.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`${creditLinkClass} text-sm leading-relaxed`}
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.45 0 0)",
                  }}
                >
                  {credit.label}
                </a>
              </li>
            ))}
          </ul>

          <section
            className="mt-8 rounded-xl border p-6 lg:p-8"
            style={{
              borderColor: "oklch(0.88 0 0)",
              background: "oklch(0.99 0 0)",
            }}
          >
            <h2 className="text-lg font-medium">Fonts</h2>
            <p
              className="mt-3 text-sm leading-relaxed"
              style={{
                fontFamily: "'DM Sans', sans-serif",
                color: "oklch(0.45 0 0)",
              }}
            >
              <a
                href="https://github.com/googlefonts/dm-fonts"
                target="_blank"
                rel="noopener noreferrer"
                className={creditLinkClass}
              >
                DM Sans
              </a>{" "}
              and{" "}
              <a
                href="https://github.com/Instrument/instrument-serif"
                target="_blank"
                rel="noopener noreferrer"
                className={creditLinkClass}
              >
                Instrument Serif
              </a>{" "}
              are distributed under the{" "}
              <a
                href={fontLicenseUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={creditLinkClass}
              >
                SIL Open Font License, Version 1.1
              </a>
              .
            </p>
          </section>
        </div>
      </main>
      <FooterSection />
    </div>
  );
}
