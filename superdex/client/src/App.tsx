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

import { Toaster } from "@/components/ui/sonner";
import NotFound from "@/pages/NotFound";
import { useEffect } from "react";
import { Route, Router, Switch, useLocation } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import RoadmapPage from "./pages/Roadmap";
import GetStartedPage from "./pages/GetStarted";
import CreditsPage from "./pages/Credits";
// @oss-disable: import { internalRoutes } from "@/internal/routes";
const internalRoutes = () => []; // @oss-enable

// Strip trailing slash so wouter's base matching works correctly
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "") || "/";

function PageViewTracker() {
  const [location] = useLocation();

  useEffect(() => {
    const gtag = (
      window as Window & {
        gtag?: (...args: unknown[]) => void;
      }
    ).gtag;

    gtag?.("event", "page_view", {
      page_title: document.title,
      page_location: window.location.href,
      page_path: window.location.pathname,
    });
  }, [location]);

  return null;
}

function AppRouter() {
  return (
    <Router base={BASE}>
      <PageViewTracker />
      <Switch>
        <Route path={"/"} component={Home} />
        <Route path={"/roadmap"} component={RoadmapPage} />
        <Route path={"/get-started"} component={GetStartedPage} />
        <Route path={"/credits"} component={CreditsPage} />
        {/* === INTERNAL-ONLY START — strip with client/src/internal/ === */}
        {internalRoutes()}
        {/* === INTERNAL-ONLY END === */}
        <Route component={NotFound} />
      </Switch>
    </Router>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="light">
        <Toaster />
        <AppRouter />
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
