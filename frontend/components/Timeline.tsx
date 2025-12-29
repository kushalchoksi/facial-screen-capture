'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { useEditorStore } from '@/stores/editorStore';
import type { Segment } from '@/types';

const TIMELINE_HEIGHT = 100;
const TRACK_HEIGHT = 40;
const TRACK_Y = 30;
const HANDLE_WIDTH = 8;
const TIME_LABEL_HEIGHT = 20;

interface DragState {
  segmentId: string;
  handle: 'start' | 'end' | 'body';
  initialX: number;
  initialStartTime: number;
  initialEndTime: number;
}

function formatTimeLabel(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function Timeline() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(800);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);
  const [hoveredHandle, setHoveredHandle] = useState<'start' | 'end' | null>(null);

  const {
    videoInfo,
    videoLoaded,
    segments,
    selectedSegmentId,
    currentTime,
    setSelectedSegmentId,
    updateSegmentTimes,
    seekTo,
  } = useEditorStore();

  const duration = videoInfo?.duration || 100;

  // Convert time to X position
  const timeToX = useCallback(
    (time: number) => (time / duration) * canvasWidth,
    [duration, canvasWidth]
  );

  // Convert X position to time
  const xToTime = useCallback(
    (x: number) => (x / canvasWidth) * duration,
    [duration, canvasWidth]
  );

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setCanvasWidth(entry.contentRect.width);
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Draw timeline
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = TIMELINE_HEIGHT * dpr;
    ctx.scale(dpr, dpr);

    // Colors
    const colors = {
      bg: '#21222c',
      track: '#44475a',
      segment: '#bd93f9',
      segmentHover: '#ff79c6',
      segmentSelected: '#bd93f9',
      handle: '#f8f8f2',
      playhead: '#8be9fd',
      text: '#6272a4',
      textLight: '#f8f8f2',
    };

    // Clear canvas
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, canvasWidth, TIMELINE_HEIGHT);

    // Draw track background
    ctx.fillStyle = colors.track;
    ctx.fillRect(0, TRACK_Y, canvasWidth, TRACK_HEIGHT);

    // Draw time markers
    const markerInterval = getMarkerInterval(duration);
    ctx.fillStyle = colors.text;
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';

    for (let t = 0; t <= duration; t += markerInterval) {
      const x = timeToX(t);

      // Time label
      ctx.fillStyle = colors.text;
      ctx.fillText(formatTimeLabel(t), x, TIME_LABEL_HEIGHT - 4);

      // Tick mark
      ctx.fillStyle = colors.text;
      ctx.fillRect(x - 0.5, TRACK_Y, 1, 4);
    }

    // Draw segments
    segments.forEach((segment) => {
      const isSelected = segment.id === selectedSegmentId;
      const isHovered = segment.id === hoveredSegment;
      const startX = timeToX(segment.start_time);
      const endX = timeToX(segment.end_time);
      const width = endX - startX;

      // Segment body
      ctx.fillStyle = isSelected
        ? colors.segmentSelected
        : isHovered
          ? colors.segmentHover
          : colors.segment;
      ctx.globalAlpha = isSelected ? 0.9 : 0.7;
      ctx.fillRect(startX, TRACK_Y + 4, width, TRACK_HEIGHT - 8);
      ctx.globalAlpha = 1;

      // Segment border
      if (isSelected || isHovered) {
        ctx.strokeStyle = colors.textLight;
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, TRACK_Y + 4, width, TRACK_HEIGHT - 8);
      }

      // Draw handles for selected/hovered segment
      if (isSelected || isHovered) {
        // Left handle
        ctx.fillStyle =
          hoveredHandle === 'start' ? colors.segmentHover : colors.handle;
        ctx.fillRect(startX - HANDLE_WIDTH / 2, TRACK_Y + 2, HANDLE_WIDTH, TRACK_HEIGHT - 4);

        // Right handle
        ctx.fillStyle =
          hoveredHandle === 'end' ? colors.segmentHover : colors.handle;
        ctx.fillRect(endX - HANDLE_WIDTH / 2, TRACK_Y + 2, HANDLE_WIDTH, TRACK_HEIGHT - 4);
      }
    });

    // Draw playhead
    if (videoLoaded) {
      const playheadX = timeToX(currentTime);

      // Playhead line
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, TIME_LABEL_HEIGHT);
      ctx.lineTo(playheadX, TIMELINE_HEIGHT);
      ctx.stroke();

      // Playhead triangle
      ctx.fillStyle = colors.playhead;
      ctx.beginPath();
      ctx.moveTo(playheadX - 6, TIME_LABEL_HEIGHT);
      ctx.lineTo(playheadX + 6, TIME_LABEL_HEIGHT);
      ctx.lineTo(playheadX, TIME_LABEL_HEIGHT + 8);
      ctx.closePath();
      ctx.fill();
    }
  }, [
    canvasWidth,
    duration,
    segments,
    selectedSegmentId,
    hoveredSegment,
    hoveredHandle,
    currentTime,
    videoLoaded,
    timeToX,
  ]);

  // Get segment and handle at position
  const getSegmentAtPosition = useCallback(
    (x: number, y: number): { segment: Segment | null; handle: 'start' | 'end' | 'body' | null } => {
      // Check if y is in track area
      if (y < TRACK_Y || y > TRACK_Y + TRACK_HEIGHT) {
        return { segment: null, handle: null };
      }

      const time = xToTime(x);

      for (const segment of segments) {
        const startX = timeToX(segment.start_time);
        const endX = timeToX(segment.end_time);

        // Check left handle
        if (Math.abs(x - startX) <= HANDLE_WIDTH) {
          return { segment, handle: 'start' };
        }

        // Check right handle
        if (Math.abs(x - endX) <= HANDLE_WIDTH) {
          return { segment, handle: 'end' };
        }

        // Check body
        if (x >= startX && x <= endX) {
          return { segment, handle: 'body' };
        }
      }

      return { segment: null, handle: null };
    },
    [segments, timeToX, xToTime]
  );

  // Mouse move handler
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (dragState) {
        // Handle dragging
        const deltaX = x - dragState.initialX;
        const deltaTime = xToTime(deltaX) - xToTime(0);

        let newStartTime = dragState.initialStartTime;
        let newEndTime = dragState.initialEndTime;

        if (dragState.handle === 'start') {
          newStartTime = Math.max(0, Math.min(dragState.initialStartTime + deltaTime, newEndTime - 0.1));
        } else if (dragState.handle === 'end') {
          newEndTime = Math.min(duration, Math.max(dragState.initialEndTime + deltaTime, newStartTime + 0.1));
        } else if (dragState.handle === 'body') {
          const segmentDuration = dragState.initialEndTime - dragState.initialStartTime;
          newStartTime = Math.max(0, dragState.initialStartTime + deltaTime);
          newEndTime = newStartTime + segmentDuration;

          if (newEndTime > duration) {
            newEndTime = duration;
            newStartTime = newEndTime - segmentDuration;
          }
        }

        // Update segment visually (will be persisted on mouseup)
        const segmentIndex = segments.findIndex((s) => s.id === dragState.segmentId);
        if (segmentIndex >= 0) {
          segments[segmentIndex] = {
            ...segments[segmentIndex],
            start_time: newStartTime,
            end_time: newEndTime,
          };
        }

        canvas.style.cursor = 'ew-resize';
      } else {
        // Handle hover
        const { segment, handle } = getSegmentAtPosition(x, y);

        if (segment) {
          setHoveredSegment(segment.id);
          setHoveredHandle(handle === 'body' ? null : handle);
          canvas.style.cursor = handle === 'body' ? 'pointer' : 'ew-resize';
        } else {
          setHoveredSegment(null);
          setHoveredHandle(null);
          canvas.style.cursor = 'pointer';
        }
      }
    },
    [dragState, getSegmentAtPosition, xToTime, duration, segments]
  );

  // Mouse down handler
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const { segment, handle } = getSegmentAtPosition(x, y);

      if (segment && handle) {
        // Start dragging
        setDragState({
          segmentId: segment.id,
          handle,
          initialX: x,
          initialStartTime: segment.start_time,
          initialEndTime: segment.end_time,
        });
        setSelectedSegmentId(segment.id);
      } else if (y >= TIME_LABEL_HEIGHT) {
        // Click on track - seek
        const time = xToTime(x);
        seekTo(time);
      }
    },
    [getSegmentAtPosition, setSelectedSegmentId, xToTime, seekTo]
  );

  // Mouse up handler
  const handleMouseUp = useCallback(async () => {
    if (dragState) {
      const segment = segments.find((s) => s.id === dragState.segmentId);
      if (segment) {
        // Persist the changes
        await updateSegmentTimes(
          dragState.segmentId,
          segment.start_time,
          segment.end_time
        );
      }
      setDragState(null);
    }
  }, [dragState, segments, updateSegmentTimes]);

  // Mouse leave handler
  const handleMouseLeave = useCallback(() => {
    setHoveredSegment(null);
    setHoveredHandle(null);
    if (dragState) {
      handleMouseUp();
    }
  }, [dragState, handleMouseUp]);

  // Global mouse up listener for drag
  useEffect(() => {
    if (dragState) {
      const handleGlobalMouseUp = () => handleMouseUp();
      window.addEventListener('mouseup', handleGlobalMouseUp);
      return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
    }
  }, [dragState, handleMouseUp]);

  return (
    <div
      ref={containerRef}
      className="w-full bg-dracula-bg-dark border-t border-dracula-bg-light"
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: TIMELINE_HEIGHT }}
        className="timeline-canvas"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
}

// Helper to determine appropriate marker interval
function getMarkerInterval(duration: number): number {
  if (duration <= 30) return 5;
  if (duration <= 60) return 10;
  if (duration <= 300) return 30;
  if (duration <= 600) return 60;
  if (duration <= 1800) return 120;
  return 300;
}
