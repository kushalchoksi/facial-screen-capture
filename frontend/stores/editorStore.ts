'use client';

import { create } from 'zustand';
import type { Segment, VideoInfo, DetectionStatus } from '@/types';
import * as api from '@/lib/api';

interface EditorState {
  // Video state
  videoInfo: VideoInfo | null;
  videoLoaded: boolean;
  currentTime: number;
  isPlaying: boolean;
  videoRef: HTMLVideoElement | null;

  // Segments state
  segments: Segment[];
  selectedSegmentId: string | null;

  // Detection state
  detectionStatus: DetectionStatus;
  isDetecting: boolean;

  // UI state
  isLoadModalOpen: boolean;
  isExporting: boolean;

  // Actions
  setVideoRef: (ref: HTMLVideoElement | null) => void;
  setVideoInfo: (info: VideoInfo) => void;
  setVideoLoaded: (loaded: boolean) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setSegments: (segments: Segment[]) => void;
  setSelectedSegmentId: (id: string | null) => void;
  setDetectionStatus: (status: DetectionStatus) => void;
  setIsDetecting: (detecting: boolean) => void;
  setLoadModalOpen: (open: boolean) => void;
  setIsExporting: (exporting: boolean) => void;

  // Complex actions
  loadVideo: (
    videoFile: File | null,
    videoPath: string | null,
    referenceFiles: File[] | null,
    referenceFolder: string | null,
    clipPadding: number
  ) => Promise<void>;
  pollDetectionStatus: () => Promise<void>;
  fetchSegments: () => Promise<void>;
  updateSegmentTimes: (
    segmentId: string,
    startTime?: number,
    endTime?: number
  ) => Promise<void>;
  toggleSegmentSelected: (segmentId: string) => Promise<void>;
  deleteSegment: (segmentId: string) => Promise<void>;
  addManualSegment: (startTime: number, endTime: number) => Promise<void>;
  exportSelectedClips: () => Promise<string[]>;
  selectAllSegments: (selected: boolean) => Promise<void>;
  seekToSegment: (segmentId: string) => void;

  // Video controls
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  seekTo: (time: number) => void;
  stepFrame: (direction: 1 | -1) => void;
  skip: (seconds: number) => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  // Initial state
  videoInfo: null,
  videoLoaded: false,
  currentTime: 0,
  isPlaying: false,
  videoRef: null,
  segments: [],
  selectedSegmentId: null,
  detectionStatus: {
    running: false,
    progress: 0,
    phase: 'idle',
    message: '',
  },
  isDetecting: false,
  isLoadModalOpen: false,
  isExporting: false,

  // Simple setters
  setVideoRef: (ref) => set({ videoRef: ref }),
  setVideoInfo: (info) => set({ videoInfo: info }),
  setVideoLoaded: (loaded) => set({ videoLoaded: loaded }),
  setCurrentTime: (time) => set({ currentTime: time }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setSegments: (segments) => set({ segments }),
  setSelectedSegmentId: (id) => set({ selectedSegmentId: id }),
  setDetectionStatus: (status) => set({ detectionStatus: status }),
  setIsDetecting: (detecting) => set({ isDetecting: detecting }),
  setLoadModalOpen: (open) => set({ isLoadModalOpen: open }),
  setIsExporting: (exporting) => set({ isExporting: exporting }),

  // Complex actions
  loadVideo: async (videoFile, videoPath, referenceFiles, referenceFolder, clipPadding) => {
    try {
      const response = await api.loadVideo(
        videoFile,
        videoPath,
        referenceFiles,
        referenceFolder,
        clipPadding
      );
      set({
        videoInfo: response.video_info,
        videoLoaded: true,
        isDetecting: true,
        segments: [],
        selectedSegmentId: null,
        isLoadModalOpen: false,
      });

      // Start polling for detection status
      get().pollDetectionStatus();
    } catch (error) {
      console.error('Failed to load video:', error);
      throw error;
    }
  },

  pollDetectionStatus: async () => {
    const poll = async () => {
      try {
        const status = await api.getDetectionStatus();
        set({ detectionStatus: status });

        if (status.running) {
          setTimeout(poll, 500);
        } else {
          set({ isDetecting: false });
          // Fetch segments when detection is complete
          if (status.phase === 'complete') {
            await get().fetchSegments();
          }
        }
      } catch (error) {
        console.error('Failed to poll status:', error);
        set({ isDetecting: false });
      }
    };
    poll();
  },

  fetchSegments: async () => {
    try {
      const { segments } = await api.getSegments();
      set({ segments });
    } catch (error) {
      console.error('Failed to fetch segments:', error);
    }
  },

  updateSegmentTimes: async (segmentId, startTime, endTime) => {
    try {
      const updates: Partial<Pick<Segment, 'start_time' | 'end_time'>> = {};
      if (startTime !== undefined) updates.start_time = startTime;
      if (endTime !== undefined) updates.end_time = endTime;

      const updated = await api.updateSegment(segmentId, updates);
      set((state) => ({
        segments: state.segments
          .map((s) => (s.id === segmentId ? updated : s))
          .sort((a, b) => a.start_time - b.start_time),
      }));
    } catch (error) {
      console.error('Failed to update segment:', error);
    }
  },

  toggleSegmentSelected: async (segmentId) => {
    const segment = get().segments.find((s) => s.id === segmentId);
    if (!segment) return;

    try {
      const updated = await api.updateSegment(segmentId, {
        selected: !segment.selected,
      });
      set((state) => ({
        segments: state.segments.map((s) => (s.id === segmentId ? updated : s)),
      }));
    } catch (error) {
      console.error('Failed to toggle segment:', error);
    }
  },

  deleteSegment: async (segmentId) => {
    try {
      await api.deleteSegment(segmentId);
      set((state) => ({
        segments: state.segments.filter((s) => s.id !== segmentId),
        selectedSegmentId:
          state.selectedSegmentId === segmentId ? null : state.selectedSegmentId,
      }));
    } catch (error) {
      console.error('Failed to delete segment:', error);
    }
  },

  addManualSegment: async (startTime, endTime) => {
    try {
      const newSegment = await api.addSegment(startTime, endTime);
      set((state) => ({
        segments: [...state.segments, newSegment].sort(
          (a, b) => a.start_time - b.start_time
        ),
      }));
    } catch (error) {
      console.error('Failed to add segment:', error);
    }
  },

  exportSelectedClips: async () => {
    const selectedIds = get()
      .segments.filter((s) => s.selected)
      .map((s) => s.id);

    if (selectedIds.length === 0) {
      throw new Error('No clips selected for export');
    }

    set({ isExporting: true });
    try {
      const result = await api.exportClips(selectedIds);
      console.log('Exported clips:', result);
      return result.exported;
    } finally {
      set({ isExporting: false });
    }
  },

  selectAllSegments: async (selected) => {
    const { segments } = get();
    try {
      await Promise.all(
        segments.map((s) => api.updateSegment(s.id, { selected }))
      );
      set((state) => ({
        segments: state.segments.map((s) => ({ ...s, selected })),
      }));
    } catch (error) {
      console.error('Failed to update segments:', error);
    }
  },

  seekToSegment: (segmentId) => {
    const segment = get().segments.find((s) => s.id === segmentId);
    if (segment) {
      get().seekTo(segment.start_time);
      set({ selectedSegmentId: segmentId });
    }
  },

  // Video controls
  play: () => {
    const { videoRef } = get();
    if (videoRef) {
      videoRef.play();
      set({ isPlaying: true });
    }
  },

  pause: () => {
    const { videoRef } = get();
    if (videoRef) {
      videoRef.pause();
      set({ isPlaying: false });
    }
  },

  togglePlayPause: () => {
    const { isPlaying } = get();
    if (isPlaying) {
      get().pause();
    } else {
      get().play();
    }
  },

  seekTo: (time) => {
    const { videoRef, videoInfo } = get();
    if (videoRef && videoInfo) {
      const clampedTime = Math.max(0, Math.min(time, videoInfo.duration));
      videoRef.currentTime = clampedTime;
      set({ currentTime: clampedTime });
    }
  },

  stepFrame: (direction) => {
    const { videoRef, videoInfo, currentTime } = get();
    if (videoRef && videoInfo) {
      const frameTime = 1 / videoInfo.fps;
      const newTime = currentTime + direction * frameTime;
      get().seekTo(newTime);
    }
  },

  skip: (seconds) => {
    const { currentTime } = get();
    get().seekTo(currentTime + seconds);
  },
}));
