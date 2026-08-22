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
 * FOOTER — Clean, minimal
 * Dark section for contrast at the bottom
 */
import { ArrowRight } from "lucide-react";
import { useLocation } from "wouter";
import { docHref } from "@/lib/docs";
import { LAYERS } from "@/lib/layers";
import { scrollToHashOnHome } from "@/lib/scroll";

type FooterLink = {
  label: string;
  href: string;
  crossSite?: boolean;
  newTab?: boolean;
};

const links: Record<string, readonly FooterLink[]> = {
  "Quick Links": [
    { label: "Roadmap", href: "/roadmap" },
    { label: "Gallery", href: "#showcase" },
    { label: "Get Started", href: "/get-started" },
  ],
  "Explore SuperDex": LAYERS.map((layer) => ({
    label: layer.title,
    href: docHref(layer.slug),
    crossSite: true,
    newTab: true,
  })),
  Resources: [
    {
      label: "GitHub",
      href: "https://github.com/facebookresearch/project_superdex",
      crossSite: true,
      newTab: true,
    },
    {
      label: "Terms of Use",
      href: "https://opensource.fb.com/legal/terms",
      crossSite: true,
      newTab: true,
    },
    {
      label: "Privacy Policy",
      href: "https://opensource.fb.com/legal/privacy",
      crossSite: true,
      newTab: true,
    },
  ],
};

const browserHref = (href: string) => {
  if (href.startsWith("/")) {
    return `${import.meta.env.BASE_URL}${href.slice(1)}`;
  }
  if (href.startsWith("#")) {
    // Footer hashes include the landing root so native modifier/new-tab navigation from a subpage opens the intended section.
    return `${import.meta.env.BASE_URL}${href}`;
  }
  return href;
};

export default function FooterSection() {
  const [location, setLocation] = useLocation();
  const isHome = location === "/";

  const handleNav = (href: string) => {
    // No-op for missing or placeholder hrefs to avoid querySelector("#") crashing.
    if (!href || href === "#") return;

    // Route links (e.g. /roadmap) navigate directly.
    if (href.startsWith("/")) {
      setLocation(href);
      return;
    }

    // Hash anchors only exist on Home. From another route (e.g. /roadmap),
    // navigate Home first, then scroll once the target section mounts.
    if (!isHome) {
      scrollToHashOnHome(href, setLocation);
      return;
    }

    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <footer style={{ background: "oklch(0.12 0 0)" }}>
      {/* CTA band */}
      <div className="border-b" style={{ borderColor: "oklch(1 0 0 / 8%)" }}>
        <div className="container py-16 lg:py-20">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8">
            <div>
              <h2
                className="display-heading mb-3"
                style={{
                  fontSize: "clamp(1.8rem, 3vw, 2.6rem)",
                  color: "oklch(0.97 0 0)",
                }}
              >
                Ready to build
                <br />
                dexterous robots?
              </h2>
              <p
                className="body-text"
                style={{
                  color: "oklch(0.55 0 0)",
                  maxWidth: "28rem",
                }}
              >
                SuperDex harnesses the power of simulation for the development
                lifecycle of dexterous AI — powered by the SuperDex Physics engine.
                Researchers and engineers are building the foundation for
                robots, starting with the hardest problem: dexterous manipulation.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3">
              <a
                href="https://github.com/facebookresearch/project_superdex"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded text-sm font-medium no-underline transition-all duration-200"
                style={{
                  background: "oklch(0.97 0 0)",
                  color: "oklch(0.12 0 0)",
                  fontFamily: "'DM Sans', sans-serif",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "oklch(0.88 0 0)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "oklch(0.97 0 0)")
                }
              >
                View on GitHub <ArrowRight size={14} />
              </a>
              <button
                onClick={() => handleNav("#overview")}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded text-sm font-medium border transition-all duration-200"
                style={{
                  borderColor: "oklch(1 0 0 / 18%)",
                  color: "oklch(0.72 0 0)",
                  fontFamily: "'DM Sans', sans-serif",
                  background: "transparent",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "oklch(1 0 0 / 35%)";
                  e.currentTarget.style.color = "oklch(0.97 0 0)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "oklch(1 0 0 / 18%)";
                  e.currentTarget.style.color = "oklch(0.72 0 0)";
                }}
              >
                Explore
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Links grid */}
      <div className="border-b" style={{ borderColor: "oklch(1 0 0 / 8%)" }}>
        <div className="container py-12">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {/* Brand */}
            <div>
              <div className="flex items-center mb-4">
                <img
                  src={`${import.meta.env.BASE_URL}img/superdex-logo-horizontal-white.png`}
                  alt="Project SuperDex"
                  className="h-8 w-auto"
                />
              </div>
              <p
                className="text-xs leading-relaxed"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
              >
                Dexterous manipulation research platform powered by the SuperDex Physics engine.
                Built by Meta Reality Labs Research.
              </p>
            </div>

            {/* Link columns */}
            {Object.entries(links).map(([category, items]) => (
              <div key={category}>
                <p
                  className="text-xs font-medium mb-4"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.55 0 0)",
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                  }}
                >
                  {category}
                </p>
                <ul className="flex flex-col gap-2.5">
                  {items.map((item) => (
                    <li key={item.label}>
                      <a
                        href={item.crossSite ? item.href : browserHref(item.href)}
                        onClick={(e) => {
                          if (item.crossSite) {
                            return;
                          }
                          if (
                            e.button !== 0 ||
                            e.metaKey ||
                            e.ctrlKey ||
                            e.shiftKey ||
                            e.altKey
                          ) {
                            return;
                          }
                          e.preventDefault();
                          handleNav(item.href);
                        }}
                        target={item.newTab ? "_blank" : undefined}
                        rel={item.newTab ? "noopener noreferrer" : undefined}
                        className="text-xs no-underline transition-colors"
                        style={{ color: "oklch(0.45 0 0)" }}
                        onMouseEnter={(e) =>
                          (e.currentTarget.style.color = "oklch(0.88 0 0)")
                        }
                        onMouseLeave={(e) =>
                          (e.currentTarget.style.color = "oklch(0.45 0 0)")
                        }
                      >
                        {item.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div className="container py-5 flex flex-col sm:flex-row items-center justify-between gap-3">
        <p
          className="text-xs"
          style={{
            fontFamily: "'DM Sans', sans-serif",
            color: "oklch(0.38 0 0)",
          }}
        >
          Copyright © {new Date().getFullYear()} Meta Platforms, Inc. Project
          SuperDex. All rights reserved.
        </p>
        <p
          className="text-xs"
          style={{
            fontFamily: "'DM Sans', sans-serif",
            color: "oklch(0.38 0 0)",
          }}
        >
          Built by Meta Reality Labs Research
        </p>
      </div>
    </footer>
  );
}
