'use client';

import { useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';
import { getVideoStreamUrl } from '@/lib/api';

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins.toString().padStart(2, '0')}:${secs
    .toString()
    .padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

export function VideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const {
    videoLoaded,
    videoInfo,
    isPlaying,
    currentTime,
    setVideoRef,
    setCurrentTime,
    setIsPlaying,
    play,
    pause,
    togglePlayPause,
    stepFrame,
    skip,
    seekTo,
  } = useEditorStore();

  // Set video ref in store
  useEffect(() => {
    setVideoRef(videoRef.current);
    return () => setVideoRef(null);
  }, [setVideoRef]);

  // Handle time updates
  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, [setCurrentTime]);

  // Handle play/pause events
  const handlePlay = useCallback(() => setIsPlaying(true), [setIsPlaying]);
  const handlePause = useCallback(() => setIsPlaying(false), [setIsPlaying]);

  // Handle video end
  const handleEnded = useCallback(() => {
    setIsPlaying(false);
    seekTo(0);
  }, [setIsPlaying, seekTo]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          togglePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          stepFrame(-1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          stepFrame(1);
          break;
        case 'KeyJ':
          e.preventDefault();
          skip(-5);
          break;
        case 'KeyL':
          e.preventDefault();
          skip(5);
          break;
        case 'Home':
          e.preventDefault();
          seekTo(0);
          break;
        case 'End':
          e.preventDefault();
          if (videoInfo) seekTo(videoInfo.duration);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlayPause, stepFrame, skip, seekTo, videoInfo]);

  return (
    <div className="flex flex-col h-full bg-dracula-bg-dark">
      {/* Video container */}
      <div className="flex-1 flex items-center justify-center p-4 min-h-0">
        {videoLoaded ? (
          <video
            ref={videoRef}
            src={getVideoStreamUrl()}
            className="max-w-full max-h-full object-contain rounded shadow-lg"
            onTimeUpdate={handleTimeUpdate}
            onPlay={handlePlay}
            onPause={handlePause}
            onEnded={handleEnded}
          />
        ) : (
          <div className="flex flex-col items-center justify-center text-dracula-comment">
            <div className="w-32 h-32 rounded-lg bg-dracula-bg-light flex items-center justify-center mb-4">
              <Play className="w-12 h-12" />
            </div>
            <p className="text-sm">Load a video to get started</p>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between px-4 py-3 bg-dracula-bg border-t border-dracula-bg-light">
        <div className="flex items-center gap-2">
          <button
            onClick={() => seekTo(0)}
            disabled={!videoLoaded}
            className="p-2 rounded hover:bg-dracula-bg-light disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Go to start (Home)"
          >
            <SkipBack className="w-4 h-4" />
          </button>

          <button
            onClick={() => stepFrame(-1)}
            disabled={!videoLoaded}
            className="p-2 rounded hover:bg-dracula-bg-light disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Previous frame (Left Arrow)"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <button
            onClick={togglePlayPause}
            disabled={!videoLoaded}
            className="p-3 rounded-full bg-dracula-purple hover:bg-dracula-pink disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Play/Pause (Space)"
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5" />
            )}
          </button>

          <button
            onClick={() => stepFrame(1)}
            disabled={!videoLoaded}
            className="p-2 rounded hover:bg-dracula-bg-light disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Next frame (Right Arrow)"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => videoInfo && seekTo(videoInfo.duration)}
            disabled={!videoLoaded}
            className="p-2 rounded hover:bg-dracula-bg-light disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Go to end (End)"
          >
            <SkipForward className="w-4 h-4" />
          </button>
        </div>

        <div className="flex items-center gap-4">
          <div className="font-mono text-sm">
            <span className="text-dracula-fg">{formatTime(currentTime)}</span>
            <span className="text-dracula-comment mx-1">/</span>
            <span className="text-dracula-comment">
              {videoInfo ? formatTime(videoInfo.duration) : '00:00.00'}
            </span>
          </div>

          {videoInfo && (
            <div className="text-xs text-dracula-comment">
              {videoInfo.fps.toFixed(2)} FPS
            </div>
          )}
        </div>

        <div className="text-xs text-dracula-comment">
          <span className="px-1.5 py-0.5 bg-dracula-bg-light rounded mr-1">Space</span>
          Play/Pause
          <span className="px-1.5 py-0.5 bg-dracula-bg-light rounded mx-1 ml-3">J/L</span>
          ±5s
          <span className="px-1.5 py-0.5 bg-dracula-bg-light rounded mx-1 ml-3">←/→</span>
          Frame
        </div>
      </div>
    </div>
  );
}
