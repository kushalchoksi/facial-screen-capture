'use client';

import { useCallback, useEffect } from 'react';
import { useEditorStore } from '@/stores/editorStore';

export function useTimeline() {
  const {
    videoInfo,
    segments,
    selectedSegmentId,
    currentTime,
    setSelectedSegmentId,
    updateSegmentTimes,
    seekTo,
  } = useEditorStore();

  const duration = videoInfo?.duration || 0;

  // Find segment at current time
  const currentSegment = segments.find(
    (s) => currentTime >= s.start_time && currentTime <= s.end_time
  );

  // Select next/previous segment
  const selectNextSegment = useCallback(() => {
    if (segments.length === 0) return;

    const currentIndex = segments.findIndex((s) => s.id === selectedSegmentId);
    const nextIndex = currentIndex < segments.length - 1 ? currentIndex + 1 : 0;
    const nextSegment = segments[nextIndex];

    setSelectedSegmentId(nextSegment.id);
    seekTo(nextSegment.start_time);
  }, [segments, selectedSegmentId, setSelectedSegmentId, seekTo]);

  const selectPreviousSegment = useCallback(() => {
    if (segments.length === 0) return;

    const currentIndex = segments.findIndex((s) => s.id === selectedSegmentId);
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : segments.length - 1;
    const prevSegment = segments[prevIndex];

    setSelectedSegmentId(prevSegment.id);
    seekTo(prevSegment.start_time);
  }, [segments, selectedSegmentId, setSelectedSegmentId, seekTo]);

  // Trim current segment to current time
  const trimStartToCurrent = useCallback(async () => {
    if (!selectedSegmentId) return;

    const segment = segments.find((s) => s.id === selectedSegmentId);
    if (!segment) return;

    if (currentTime > segment.start_time && currentTime < segment.end_time) {
      await updateSegmentTimes(selectedSegmentId, currentTime, undefined);
    }
  }, [selectedSegmentId, segments, currentTime, updateSegmentTimes]);

  const trimEndToCurrent = useCallback(async () => {
    if (!selectedSegmentId) return;

    const segment = segments.find((s) => s.id === selectedSegmentId);
    if (!segment) return;

    if (currentTime > segment.start_time && currentTime < segment.end_time) {
      await updateSegmentTimes(selectedSegmentId, undefined, currentTime);
    }
  }, [selectedSegmentId, segments, currentTime, updateSegmentTimes]);

  // Keyboard shortcuts for timeline navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.code) {
        case 'BracketLeft':
          e.preventDefault();
          selectPreviousSegment();
          break;
        case 'BracketRight':
          e.preventDefault();
          selectNextSegment();
          break;
        case 'KeyI':
          e.preventDefault();
          trimStartToCurrent();
          break;
        case 'KeyO':
          e.preventDefault();
          trimEndToCurrent();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectPreviousSegment, selectNextSegment, trimStartToCurrent, trimEndToCurrent]);

  return {
    duration,
    currentSegment,
    selectNextSegment,
    selectPreviousSegment,
    trimStartToCurrent,
    trimEndToCurrent,
  };
}
