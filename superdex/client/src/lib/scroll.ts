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
 * Cross-page in-page navigation helper.
 *
 * Section anchors (e.g. "#physics") only exist on the Home route. A link on
 * another route (e.g. /roadmap) that targets one must navigate Home and then
 * scroll once the section mounts.
 *
 * The polling interval here is intentionally NOT tied to a React component's
 * lifecycle: the component that initiated the click unmounts as part of this
 * very navigation, so a lifecycle-bound interval (cleared on unmount) would be
 * killed before it could run. This standalone interval self-terminates once the
 * target is found or after a bounded number of attempts, so it does not leak.
 */
export function scrollToHashOnHome(
  hash: string,
  navigate: (to: string) => void,
): void {
  navigate("/");
  let attempts = 0;
  const poll = setInterval(() => {
    attempts++;
    const el = document.querySelector(hash);
    if (el) {
      clearInterval(poll);
      el.scrollIntoView({ behavior: "smooth" });
    } else if (attempts >= 40) {
      clearInterval(poll);
    }
  }, 50);
}
