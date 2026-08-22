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
 * GET STARTED PAGE — standalone tab presenting the layer entry points.
 * Mirrors the roadmap-tab pattern: public Navbar + fixed-nav spacer + Footer.
 */
import { useEffect } from "react";
import Navbar from "@/components/Navbar";
import GetStarted from "@/components/GetStarted";
import FooterSection from "@/components/FooterSection";

const NAV_HEIGHT = 64;

export default function GetStartedPage() {
  // Open the tab at the top rather than inheriting the prior page's scroll.
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
      <main className="flex-1">
        <GetStarted />
      </main>
      <FooterSection />
    </div>
  );
}
