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
 * HOME PAGE — Project SuperDex
 * Assembles all sections in order with clean light theme
 */
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import Overview from "@/components/Overview";
import LayerHighlights from "@/components/LayerHighlights";
import SimulationShowcase from "@/components/SimulationShowcase";
import FooterSection from "@/components/FooterSection";

export default function Home() {
  return (
    <div
      className="min-h-screen"
      style={{ background: "oklch(1 0 0)" }}
    >
      <Navbar />
      <main>
        <HeroSection />
        <Overview />
        <SimulationShowcase />
        <LayerHighlights />
      </main>
      <FooterSection />
    </div>
  );
}
