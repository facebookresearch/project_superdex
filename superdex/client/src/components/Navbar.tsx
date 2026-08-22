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
 * NAVBAR — Clean, minimal top navigation
 * Behavior: transparent on hero, solid warm-white on scroll
 */
import { useCallback, useState, useEffect, useRef } from "react";
import type {
  FocusEvent as ReactFocusEvent,
  KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronDown, Menu, X } from "lucide-react";
import { useLocation } from "wouter";
import { docHref } from "@/lib/docs";
import type { ProjectSite } from "@/lib/docs";
import { scrollToHashOnHome } from "@/lib/scroll";
// @oss-disable: import { INTERNAL_BASE, INTERNAL_ENABLED } from "@/internal/config";
const INTERNAL_BASE = "/internal"; // @oss-enable
const INTERNAL_ENABLED = false; // @oss-enable

const navLinks = [
  { label: "Overview", href: "#overview" },
  { label: "Gallery", href: "#showcase" },
];

const layerLinks: readonly { label: string; site: ProjectSite }[] = [
  { label: "Physics", site: "physics" },
  { label: "Robotics", site: "robotics" },
  { label: "Studio", site: "studio" },
  { label: "Lab", site: "lab" },
];

type ExploreMenu = "desktop" | "mobile";
type ExploreFocusPosition = "first" | "last";

function focusMenuItem(
  menu: HTMLDivElement,
  position: ExploreFocusPosition,
) {
  const items = menu.querySelectorAll<HTMLAnchorElement>('[role="menuitem"]');
  const item = position === "first" ? items[0] : items[items.length - 1];
  item?.focus();
}

export default function Navbar() {
  const reduceMotion = useReducedMotion();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [exploreMenu, setExploreMenu] = useState<ExploreMenu | null>(null);
  const [location, setLocation] = useLocation();
  const isHome = location === "/";
  const toggleButtonRef = useRef<HTMLButtonElement>(null);
  const mobileMenuRef = useRef<HTMLDivElement>(null);
  const desktopExploreRef = useRef<HTMLDivElement>(null);
  const mobileExploreRef = useRef<HTMLDivElement>(null);
  const desktopExploreButtonRef = useRef<HTMLButtonElement>(null);
  const mobileExploreButtonRef = useRef<HTMLButtonElement>(null);
  const desktopExploreMenuRef = useRef<HTMLDivElement | null>(null);
  const mobileExploreMenuRef = useRef<HTMLDivElement>(null);
  const pendingDesktopExploreFocusRef = useRef<ExploreFocusPosition | null>(
    null,
  );

  const setDesktopExploreMenuRef = useCallback((menu: HTMLDivElement | null) => {
    desktopExploreMenuRef.current = menu;
    const pendingFocus = pendingDesktopExploreFocusRef.current;
    if (!menu || !pendingFocus) return;

    pendingDesktopExploreFocusRef.current = null;
    focusMenuItem(menu, pendingFocus);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Mobile menu: focus the first link on open, and return focus to the toggle on close.
  useEffect(() => {
    if (!menuOpen) return;
    // Focus the first focusable element inside the mobile menu after it mounts.
    const focusTimer = window.setTimeout(() => {
      const firstFocusable = mobileMenuRef.current?.querySelector<HTMLElement>(
        "a, button",
      );
      firstFocusable?.focus();
    }, 0);
    return () => {
      window.clearTimeout(focusTimer);
      // Return focus to the toggle button when the menu closes.
      toggleButtonRef.current?.focus();
    };
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen && !exploreMenu) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;

      if (exploreMenu) {
        event.preventDefault();
        const trigger =
          exploreMenu === "desktop"
            ? desktopExploreButtonRef.current
            : mobileExploreButtonRef.current;
        setExploreMenu(null);
        window.requestAnimationFrame(() => trigger?.focus());
        return;
      }

      setMenuOpen(false);
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [exploreMenu, menuOpen]);

  useEffect(() => {
    if (!exploreMenu) return;

    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      const activeExplore =
        exploreMenu === "desktop"
          ? desktopExploreRef.current
          : mobileExploreRef.current;
      if (!activeExplore?.contains(target)) {
        setExploreMenu(null);
      }
    };

    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [exploreMenu]);

  const handleNav = (href: string) => {
    setExploreMenu(null);
    setMenuOpen(false);
    if (!isHome) {
      // Anchors live on Home; navigate there, then scroll once it mounts.
      scrollToHashOnHome(href, setLocation);
      return;
    }
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  const handleComponentNav = (href: string) => {
    setExploreMenu(null);
    setMenuOpen(false);
    setLocation(href);
  };

  const focusExploreItem = (
    menu: ExploreMenu,
    position: ExploreFocusPosition,
  ) => {
    if (menu === "desktop") {
      const mountedMenu = desktopExploreMenuRef.current;
      if (exploreMenu === "desktop" && mountedMenu) {
        focusMenuItem(mountedMenu, position);
      } else {
        pendingDesktopExploreFocusRef.current = position;
      }
      return;
    }

    window.requestAnimationFrame(() => {
      const mountedMenu = mobileExploreMenuRef.current;
      if (mountedMenu) focusMenuItem(mountedMenu, position);
    });
  };

  const toggleExploreMenu = (menu: ExploreMenu, focusOnOpen = true) => {
    if (exploreMenu === menu) {
      if (menu === "desktop") pendingDesktopExploreFocusRef.current = null;
      setExploreMenu(null);
      return;
    }
    if (menu === "desktop" && !focusOnOpen) {
      pendingDesktopExploreFocusRef.current = null;
    }
    setExploreMenu(menu);
    if (focusOnOpen) focusExploreItem(menu, "first");
  };

  const handleExploreTriggerKeyDown = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
    menu: ExploreMenu,
  ) => {
    const opensDesktopMenu =
      menu === "desktop" &&
      (event.key === "Enter" ||
        event.key === " " ||
        event.key === "Spacebar");
    if (
      event.key !== "ArrowDown" &&
      event.key !== "ArrowUp" &&
      !opensDesktopMenu
    ) {
      return;
    }
    event.preventDefault();
    setExploreMenu(menu);
    focusExploreItem(menu, event.key === "ArrowUp" ? "last" : "first");
  };

  const handleExploreMenuKeyDown = (
    event: ReactKeyboardEvent<HTMLDivElement>,
  ) => {
    const activeItem =
      document.activeElement instanceof HTMLAnchorElement &&
      document.activeElement.getAttribute("role") === "menuitem" &&
      event.currentTarget.contains(document.activeElement)
        ? document.activeElement
        : null;

    if ((event.key === " " || event.key === "Spacebar") && activeItem) {
      event.preventDefault();
      activeItem.click();
      return;
    }

    const items = Array.from(
      event.currentTarget.querySelectorAll<HTMLAnchorElement>(
        '[role="menuitem"]',
      ),
    );
    if (items.length === 0) return;

    const currentIndex = activeItem ? items.indexOf(activeItem) : -1;
    let nextIndex: number | null = null;
    if (event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % items.length;
    } else if (event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + items.length) % items.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = items.length - 1;
    }

    if (nextIndex === null) return;
    event.preventDefault();
    items[nextIndex]?.focus();
  };

  const handleExploreBlur = (event: ReactFocusEvent<HTMLDivElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setExploreMenu(null);
    }
  };

  // Always-solid background on non-home routes.
  const showSolid = scrolled || !isHome;

  return (
    <>
      <header
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          showSolid
            ? "bg-[oklch(0.99_0_0)] border-b border-[oklch(0.88_0_0)]"
            : "bg-transparent"
        }`}
      >
        <div className="container xl:max-w-[1440px]">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <a
              href={import.meta.env.BASE_URL}
              className="flex items-center no-underline"
              onClick={(e) => {
                e.preventDefault();
                setExploreMenu(null);
                setMenuOpen(false);
                if (isHome) {
                  window.scrollTo({ top: 0, behavior: "smooth" });
                } else {
                  setLocation("/");
                }
              }}
            >
              <img
                src={`${import.meta.env.BASE_URL}img/superdex-logo-horizontal.png`}
                alt="Project SuperDex"
                className="h-8 w-auto"
              />
            </a>

            {/* Desktop nav */}
            <nav aria-label="Main navigation" className="hidden xl:flex items-center gap-5">
              {navLinks.map((link) => (
                <button
                  key={link.label}
                  onClick={() => handleNav(link.href)}
                  className="text-[0.82rem] font-medium transition-colors"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.45 0 0)",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.color = "oklch(0.15 0 0)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.color = "oklch(0.45 0 0)")
                  }
                >
                  {link.label}
                </button>
              ))}

              <div
                ref={desktopExploreRef}
                className="relative flex items-center"
                onBlur={handleExploreBlur}
              >
                <button
                  ref={desktopExploreButtonRef}
                  type="button"
                  aria-expanded={exploreMenu === "desktop"}
                  aria-haspopup="menu"
                  aria-controls="desktop-explore-menu"
                  onClick={() => toggleExploreMenu("desktop", false)}
                  onKeyDown={(event) =>
                    handleExploreTriggerKeyDown(event, "desktop")
                  }
                  className="inline-flex items-center gap-1 text-[0.82rem] font-medium transition-colors"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.45 0 0)",
                  }}
                  onMouseEnter={(event) =>
                    (event.currentTarget.style.color = "oklch(0.15 0 0)")
                  }
                  onMouseLeave={(event) =>
                    (event.currentTarget.style.color = "oklch(0.45 0 0)")
                  }
                >
                  Explore SuperDex
                  <ChevronDown
                    aria-hidden="true"
                    size={14}
                    className={`transition-transform motion-reduce:transition-none ${
                      exploreMenu === "desktop" ? "rotate-180" : ""
                    }`}
                  />
                </button>

                <AnimatePresence>
                  {exploreMenu === "desktop" && (
                    <motion.div
                      ref={setDesktopExploreMenuRef}
                      id="desktop-explore-menu"
                      role="menu"
                      aria-label="Explore SuperDex"
                      initial={reduceMotion ? false : { opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={
                        reduceMotion
                          ? { opacity: 1, y: 0 }
                          : { opacity: 0, y: -6 }
                      }
                      transition={{ duration: reduceMotion ? 0 : 0.15 }}
                      onKeyDown={handleExploreMenuKeyDown}
                      className="absolute left-0 top-full z-50 mt-3 min-w-40 overflow-hidden rounded-md border py-1 shadow-lg"
                      style={{
                        background: "oklch(0.99 0 0)",
                        borderColor: "oklch(0.88 0 0)",
                      }}
                    >
                      {layerLinks.map((link) => (
                        <a
                          key={link.site}
                          href={docHref(link.site)}
                          target="_blank"
                          rel="noopener noreferrer"
                          role="menuitem"
                          tabIndex={-1}
                          onClick={() => setExploreMenu(null)}
                          className="block px-4 py-2.5 text-[0.82rem] no-underline transition-colors focus-visible:outline-2 focus-visible:outline-offset-[-2px]"
                          style={{
                            fontFamily: "'DM Sans', sans-serif",
                            color: "oklch(0.35 0 0)",
                          }}
                          onMouseEnter={(event) =>
                            (event.currentTarget.style.background =
                              "oklch(0.95 0 0)")
                          }
                          onMouseLeave={(event) =>
                            (event.currentTarget.style.background =
                              "transparent")
                          }
                        >
                          {link.label}
                        </a>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Get Started — routes to its own tab */}
              <button
                onClick={() => handleComponentNav("/get-started")}
                className="text-[0.82rem] font-medium transition-colors"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = "oklch(0.15 0 0)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = "oklch(0.45 0 0)")
                }
              >
                Get Started
              </button>

              {/* Roadmap — routes to its own tab, sits at the right of the nav */}
              <button
                onClick={() => handleComponentNav("/roadmap")}
                className="text-[0.82rem] font-medium transition-colors"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.45 0 0)",
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.color = "oklch(0.15 0 0)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.color = "oklch(0.45 0 0)")
                }
              >
                Roadmap
              </button>

              {/* === INTERNAL-ONLY START — strip with client/src/internal/ === */}
              {INTERNAL_ENABLED && (
                <button
                  onClick={() => handleComponentNav(INTERNAL_BASE)}
                  className="text-[0.82rem] font-medium transition-colors inline-flex items-center gap-1.5"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.45 0 0)",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.color = "oklch(0.15 0 0)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.color = "oklch(0.45 0 0)")
                  }
                >
                  Internal
                </button>
              )}
              {/* === INTERNAL-ONLY END === */}
            </nav>

            {/* CTAs */}
            <div className="hidden xl:flex items-center gap-3">
              <a
                href="https://github.com/facebookresearch/project_superdex"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded text-[0.82rem] font-medium no-underline transition-all duration-200"
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
                GitHub
              </a>
            </div>

            {/* Mobile toggle */}
            <button
              ref={toggleButtonRef}
              className="xl:hidden"
              style={{ color: "oklch(0.35 0 0)" }}
              onClick={() => {
                setExploreMenu(null);
                setMenuOpen(!menuOpen);
              }}
              aria-label="Toggle navigation menu"
              aria-expanded={menuOpen}
              aria-controls="mobile-menu"
            >
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile menu */}
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            id="mobile-menu"
            ref={mobileMenuRef}
            initial={reduceMotion ? false : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={
              reduceMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: -8 }
            }
            transition={{ duration: reduceMotion ? 0 : 0.18 }}
            className="fixed top-16 left-0 right-0 z-40 max-h-[calc(100dvh-4rem)] overflow-y-auto border-b xl:hidden"
            style={{
              background: "oklch(0.99 0 0)",
              borderColor: "oklch(0.88 0 0)",
            }}
          >
            <div className="container py-4 flex flex-col gap-1">
              {navLinks.map((link) => (
                <button
                  key={link.label}
                  onClick={() => handleNav(link.href)}
                  className="text-left py-3 text-sm border-b transition-colors"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.35 0 0)",
                    borderColor: "oklch(0.92 0 0)",
                  }}
                >
                  {link.label}
                </button>
              ))}
              <div ref={mobileExploreRef} onBlur={handleExploreBlur}>
                <button
                  ref={mobileExploreButtonRef}
                  type="button"
                  aria-expanded={exploreMenu === "mobile"}
                  aria-haspopup="menu"
                  aria-controls="mobile-explore-menu"
                  onClick={() => toggleExploreMenu("mobile")}
                  onKeyDown={(event) =>
                    handleExploreTriggerKeyDown(event, "mobile")
                  }
                  className="flex w-full items-center justify-between border-b py-3 text-left text-sm transition-colors"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.35 0 0)",
                    borderColor: "oklch(0.92 0 0)",
                  }}
                >
                  Explore SuperDex
                  <ChevronDown
                    aria-hidden="true"
                    size={16}
                    className={`transition-transform motion-reduce:transition-none ${
                      exploreMenu === "mobile" ? "rotate-180" : ""
                    }`}
                  />
                </button>

                <AnimatePresence>
                  {exploreMenu === "mobile" && (
                    <motion.div
                      ref={mobileExploreMenuRef}
                      id="mobile-explore-menu"
                      role="menu"
                      aria-label="Explore SuperDex"
                      initial={
                        reduceMotion ? false : { opacity: 0, height: 0 }
                      }
                      animate={{ opacity: 1, height: "auto" }}
                      exit={
                        reduceMotion
                          ? { opacity: 1, height: "auto" }
                          : { opacity: 0, height: 0 }
                      }
                      transition={{ duration: reduceMotion ? 0 : 0.15 }}
                      onKeyDown={handleExploreMenuKeyDown}
                      className="overflow-hidden border-b"
                      style={{ borderColor: "oklch(0.92 0 0)" }}
                    >
                      {layerLinks.map((link) => (
                        <a
                          key={link.site}
                          href={docHref(link.site)}
                          target="_blank"
                          rel="noopener noreferrer"
                          role="menuitem"
                          tabIndex={-1}
                          onClick={() => {
                            setExploreMenu(null);
                            setMenuOpen(false);
                          }}
                          className="block py-2.5 pl-5 text-sm no-underline transition-colors focus-visible:outline-2 focus-visible:outline-offset-[-2px]"
                          style={{
                            fontFamily: "'DM Sans', sans-serif",
                            color: "oklch(0.4 0 0)",
                          }}
                        >
                          {link.label}
                        </a>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              <button
                onClick={() => handleComponentNav("/get-started")}
                className="text-left py-3 text-sm border-b transition-colors"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.35 0 0)",
                  borderColor: "oklch(0.92 0 0)",
                }}
              >
                Get Started
              </button>
              <button
                onClick={() => handleComponentNav("/roadmap")}
                className="text-left py-3 text-sm border-b transition-colors"
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  color: "oklch(0.35 0 0)",
                  borderColor: "oklch(0.92 0 0)",
                }}
              >
                Roadmap
              </button>
              {/* === INTERNAL-ONLY START — strip with client/src/internal/ === */}
              {INTERNAL_ENABLED && (
                <button
                  onClick={() => handleComponentNav(INTERNAL_BASE)}
                  className="text-left py-3 text-sm border-b transition-colors"
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    color: "oklch(0.35 0 0)",
                    borderColor: "oklch(0.92 0 0)",
                  }}
                >
                  Internal
                </button>
              )}
              {/* === INTERNAL-ONLY END === */}
              <a
                href="https://github.com/facebookresearch/project_superdex"
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => {
                  setExploreMenu(null);
                  setMenuOpen(false);
                }}
                className="mt-3 px-4 py-2.5 rounded text-sm font-medium text-center no-underline"
                style={{
                  background: "oklch(0.15 0 0)",
                  color: "oklch(0.97 0 0)",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              >
                GitHub →
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
