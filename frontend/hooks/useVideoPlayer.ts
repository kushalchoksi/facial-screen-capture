'use client';

import { useEffect, useCallback } from 'react';
import { useEditorStore } from '@/stores/editorStore';

export function useVideoPlayer() {
  const {
    videoRef,
    videoInfo,
    isPlaying,
    currentTime,
    play,
    pause,
    togglePlayPause,
    seekTo,
    stepFrame,
    skip,
  } = useEditorStore();

  // Sync video time updates
  useEffect(() => {
    if (!videoRef) return;

    const handleTimeUpdate = () => {
      useEditorStore.getState().setCurrentTime(videoRef.currentTime);
    };

    const handlePlay = () => {
      useEditorStore.getState().setIsPlaying(true);
    };

    const handlePause = () => {
      useEditorStore.getState().setIsPlaying(false);
    };

    videoRef.addEventListener('timeupdate', handleTimeUpdate);
    videoRef.addEventListener('play', handlePlay);
    videoRef.addEventListener('pause', handlePause);

    return () => {
      videoRef.removeEventListener('timeupdate', handleTimeUpdate);
      videoRef.removeEventListener('play', handlePlay);
      videoRef.removeEventListener('pause', handlePause);
    };
  }, [videoRef]);

  // Seek to specific time with validation
  const seekToTime = useCallback(
    (time: number) => {
      if (!videoInfo) return;
      const clampedTime = Math.max(0, Math.min(time, videoInfo.duration));
      seekTo(clampedTime);
    },
    [videoInfo, seekTo]
  );

  // Get current playback progress (0-1)
  const progress = videoInfo ? currentTime / videoInfo.duration : 0;

  // Get formatted time strings
  const formatTime = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs
      .toString()
      .padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  }, []);

  const currentTimeFormatted = formatTime(currentTime);
  const durationFormatted = videoInfo ? formatTime(videoInfo.duration) : '00:00.00';

  return {
    videoRef,
    videoInfo,
    isPlaying,
    currentTime,
    progress,
    currentTimeFormatted,
    durationFormatted,
    play,
    pause,
    togglePlayPause,
    seekTo: seekToTime,
    stepFrame,
    skip,
  };
}
