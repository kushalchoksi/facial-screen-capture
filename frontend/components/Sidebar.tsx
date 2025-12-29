'use client';

import { Plus, CheckSquare, Square } from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';
import { ClipItem } from './ClipItem';

export function Sidebar() {
  const {
    segments,
    videoLoaded,
    videoInfo,
    currentTime,
    selectAllSegments,
    addManualSegment,
  } = useEditorStore();

  const allSelected = segments.length > 0 && segments.every((s) => s.selected);
  const noneSelected = segments.every((s) => !s.selected);

  const handleSelectAll = () => {
    selectAllSegments(!allSelected);
  };

  const handleAddManual = () => {
    if (!videoInfo) return;

    const duration = Math.min(5, videoInfo.duration - currentTime);
    if (duration <= 0) {
      alert('Cannot add clip at end of video');
      return;
    }

    addManualSegment(currentTime, currentTime + duration);
  };

  return (
    <aside className="w-72 flex flex-col bg-dracula-bg border-r border-dracula-bg-light">
      <div className="p-3 border-b border-dracula-bg-light">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold text-sm text-dracula-fg">
            Detected Clips
          </h2>
          <span className="text-xs text-dracula-comment">
            {segments.length} clip{segments.length !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleSelectAll}
            disabled={segments.length === 0}
            className="flex items-center gap-1.5 px-2 py-1 text-xs rounded bg-dracula-bg-light hover:bg-dracula-comment disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {allSelected ? (
              <CheckSquare className="w-3.5 h-3.5" />
            ) : (
              <Square className="w-3.5 h-3.5" />
            )}
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>

          <button
            onClick={handleAddManual}
            disabled={!videoLoaded}
            className="flex items-center gap-1.5 px-2 py-1 text-xs rounded bg-dracula-bg-light hover:bg-dracula-comment disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Add clip at current position"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Clip
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {segments.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-dracula-comment text-sm p-4 text-center">
            {videoLoaded ? (
              <>
                <p className="mb-2">No clips detected yet</p>
                <p className="text-xs">
                  Wait for detection to complete or add clips manually
                </p>
              </>
            ) : (
              <>
                <p className="mb-2">No video loaded</p>
                <p className="text-xs">
                  Load a video and reference images to detect clips
                </p>
              </>
            )}
          </div>
        ) : (
          segments.map((segment, index) => (
            <ClipItem key={segment.id} segment={segment} index={index} />
          ))
        )}
      </div>
    </aside>
  );
}
