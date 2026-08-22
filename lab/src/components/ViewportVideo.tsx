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

import React, {useEffect, useRef, useState} from 'react';

type ViewportVideoProps = {
  src: string;
  poster: string;
  ariaLabel: string;
  className?: string;
  onPlaybackTime?: (currentTimeSeconds: number) => void;
};

export default function ViewportVideo({
  src,
  poster,
  ariaLabel,
  className,
  onPlaybackTime,
}: ViewportVideoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const playbackTimeHandlerRef = useRef(onPlaybackTime);
  const [shouldLoad, setShouldLoad] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);
  const tracksPlaybackTime = onPlaybackTime != null;

  useEffect(() => {
    playbackTimeHandlerRef.current = onPlaybackTime;
  }, [onPlaybackTime]);

  useEffect(() => {
    const video = videoRef.current;
    if (
      !video ||
      !tracksPlaybackTime ||
      typeof video.requestVideoFrameCallback !== 'function'
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
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const sync = () => {
      setReduceMotion(mediaQuery.matches);
      if (mediaQuery.matches) videoRef.current?.pause();
    };

    sync();
    mediaQuery.addEventListener('change', sync);
    return () => mediaQuery.removeEventListener('change', sync);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (typeof IntersectionObserver === 'undefined') {
      setShouldLoad(true);
      setIsVisible(true);
      return;
    }

    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          setShouldLoad(true);
          setIsVisible(true);
        } else {
          setIsVisible(false);
          video.pause();
        }
      }
    });

    observer.observe(video);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (reduceMotion || !isVisible) {
      video.pause();
      return;
    }

    if (shouldLoad && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      void video.play().catch(() => {});
    }
  }, [isVisible, reduceMotion, shouldLoad]);

  const handleCanPlay = () => {
    const video = videoRef.current;
    if (!video || reduceMotion || !isVisible) return;
    void video.play().catch(() => {});
  };

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (
      !video ||
      !tracksPlaybackTime ||
      typeof video.requestVideoFrameCallback === 'function'
    ) {
      return;
    }
    playbackTimeHandlerRef.current?.(video.currentTime);
  };

  return (
    <video
      ref={videoRef}
      src={shouldLoad ? src : undefined}
      poster={poster}
      autoPlay={!reduceMotion && isVisible}
      loop
      muted
      playsInline
      preload={shouldLoad ? 'metadata' : 'none'}
      onCanPlay={handleCanPlay}
      onTimeUpdate={tracksPlaybackTime ? handleTimeUpdate : undefined}
      role="img"
      aria-label={ariaLabel}
      className={className}
    />
  );
}
