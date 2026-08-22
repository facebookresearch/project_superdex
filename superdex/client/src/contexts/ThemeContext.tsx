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

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

type Theme = "light" | "dark" | "system";

const VALID_THEMES: readonly Theme[] = ["light", "dark", "system"];
const THEME_STORAGE_KEY = "theme";

function isValidTheme(value: unknown): value is Theme {
  return (
    typeof value === "string" &&
    (VALID_THEMES as readonly string[]).includes(value)
  );
}

function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme === "system") {
    if (
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return "dark";
    }
    return "light";
  }
  return theme;
}

interface ThemeContextType {
  theme: "light" | "dark";
  toggleTheme?: () => void;
  switchable: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
  switchable?: boolean;
}

export function ThemeProvider({
  children,
  defaultTheme = "light",
  switchable = false,
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    if (switchable) {
      try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (isValidTheme(stored)) {
          return resolveTheme(stored);
        }
      } catch {}
    }
    // First-paint value must match the inline FOUC script in index.html,
    // which renders as light (no `dark` class) whenever localStorage is
    // empty or invalid. The user-supplied `defaultTheme` is applied after
    // mount via the effect below, so call sites that pass no defaultTheme
    // (or pass "light") still work identically.
    return "light";
  });

  // After mount, apply the user-supplied defaultTheme when no valid theme
  // has been persisted yet. This keeps `defaultTheme !== "light"` working
  // for callers without disagreeing with the FOUC script on first paint.
  useEffect(() => {
    if (switchable) {
      try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (isValidTheme(stored)) {
          return;
        }
      } catch {}
    }
    setTheme(resolveTheme(defaultTheme));
    // Run once on mount; defaultTheme/switchable are provider-level inputs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }

    if (switchable) {
      try {
        localStorage.setItem(THEME_STORAGE_KEY, theme);
      } catch {}
    }
  }, [theme, switchable]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === "light" ? "dark" : "light"));
  }, []);

  const value = useMemo<ThemeContextType>(
    () => ({
      theme,
      toggleTheme: switchable ? toggleTheme : undefined,
      switchable,
    }),
    [theme, switchable, toggleTheme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return context;
}
