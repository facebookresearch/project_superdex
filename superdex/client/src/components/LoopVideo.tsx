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

import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactEventHandler } from "react";

type LoopVideoProps = {
  src: string;
  poster: string;
  ariaLabel: string;
  reducedMotion: boolean;
  className?: string;
  style?: CSSProperties;
  onLoadedMetadata?: ReactEventHandler<HTMLVideoElement>;
  onTimeUpdate?: ReactEventHandler<HTMLVideoElement>;
  onPlaybackTime?: (currentTimeSeconds: number) => void;
};

export default function LoopVideo({
  src,
  poster,
  ariaLabel,
  reducedMotion,
  className,
  style,
  onLoadedMetadata,
  onTimeUpdate,
  onPlaybackTime,
}: LoopVideoProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const playbackTimeHandlerRef = useRef(onPlaybackTime);
  const [shouldLoad, setShouldLoad] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const tracksPlaybackTime = onPlaybackTime != null;

  useEffect(() => {
    playbackTimeHandlerRef.current = onPlaybackTime;
  }, [onPlaybackTime]);

  useEffect(() => {
    const video = videoRef.current;
    if (
      !video ||
      !tracksPlaybackTime ||
      typeof video.requestVideoFrameCallback !== "function"
    ) {
      return;
    }

    let callbackId = 0;
    const handleVideoFrame: VideoFrameRequestCallback = (_now, metadata) => {
      playbackTimeHandlerRef.current?.(metadata.mediaTime);
      callbackId = video.requestVideoFrameCallback(handleVideoFrame);
    };

    callbackId = video.requestVideoFrameCallback(handleVideoFrame);
    return () => video.cancelVideoFrameCallback(callbackId);
  }, [tracksPlaybackTime]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (typeof IntersectionObserver === "undefined") {
      setShouldLoad(true);
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setShouldLoad(true);
            setIsVisible(true);
          } else {
            setIsVisible(false);
            video.pause();
          }
        }
      },
      { threshold: 0.25 },
    );
    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (reducedMotion || !isVisible) {
      video.pause();
      return;
    }
    if (shouldLoad && video.readyState >= 2) {
      void video.play().catch(() => {
        /* Autoplay may be blocked by the browser. */
      });
    }
  }, [isVisible, reducedMotion, shouldLoad]);

  const handleCanPlay = () => {
    const video = videoRef.current;
    if (!video || reducedMotion || !isVisible) return;
    void video.play().catch(() => {
      /* Autoplay may be blocked by the browser. */
    });
  };

  const handleTimeUpdate: ReactEventHandler<HTMLVideoElement> = (event) => {
    onTimeUpdate?.(event);
    if (
      tracksPlaybackTime &&
      typeof event.currentTarget.requestVideoFrameCallback !== "function"
    ) {
      playbackTimeHandlerRef.current?.(event.currentTarget.currentTime);
    }
  };

  return (
    <video
      ref={videoRef}
      src={shouldLoad ? src : undefined}
      poster={poster}
      autoPlay={!reducedMotion && isVisible}
      loop
      muted
      playsInline
      preload={shouldLoad ? "metadata" : "none"}
      onCanPlay={handleCanPlay}
      onLoadedMetadata={onLoadedMetadata}
      onTimeUpdate={
        onTimeUpdate || tracksPlaybackTime ? handleTimeUpdate : undefined
      }
      role="img"
      aria-label={ariaLabel}
      className={className ?? "block h-full w-full object-cover"}
      style={style}
    />
  );
}
