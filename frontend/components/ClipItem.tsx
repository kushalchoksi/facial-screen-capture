'use client';

import { Trash2, Check, Square } from 'lucide-react';
import type { Segment } from '@/types';
import { useEditorStore } from '@/stores/editorStore';

interface ClipItemProps {
  segment: Segment;
  index: number;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}m ${secs}s`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function ClipItem({ segment, index }: ClipItemProps) {
  const {
    selectedSegmentId,
    seekToSegment,
    toggleSegmentSelected,
    deleteSegment,
  } = useEditorStore();

  const isSelected = selectedSegmentId === segment.id;
  const duration = segment.end_time - segment.start_time;

  const handleClick = () => {
    seekToSegment(segment.id);
  };

  const handleCheckboxClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleSegmentSelected(segment.id);
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Delete this clip?')) {
      deleteSegment(segment.id);
    }
  };

  return (
    <div
      onClick={handleClick}
      className={`
        clip-item p-3 rounded-lg cursor-pointer border-2 transition-all
        ${
          isSelected
            ? 'bg-dracula-purple/20 border-dracula-purple'
            : 'bg-dracula-bg-dark border-transparent hover:bg-dracula-bg-light hover:border-dracula-comment/30'
        }
      `}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={handleCheckboxClick}
            className={`
              flex-shrink-0 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors
              ${
                segment.selected
                  ? 'bg-dracula-purple border-dracula-purple'
                  : 'border-dracula-comment hover:border-dracula-purple'
              }
            `}
          >
            {segment.selected && <Check className="w-3 h-3" />}
          </button>

          <div className="min-w-0">
            <div className="font-medium text-sm truncate">
              Clip {index + 1}
            </div>
            <div className="text-xs text-dracula-comment">
              {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <span className="text-xs text-dracula-cyan font-mono">
            {formatDuration(duration)}
          </span>
          <button
            onClick={handleDeleteClick}
            className="p-1 rounded hover:bg-dracula-red/20 text-dracula-comment hover:text-dracula-red transition-colors"
            title="Delete clip"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {segment.match_count > 0 && (
        <div className="mt-2 flex items-center gap-3 text-xs text-dracula-comment">
          <span>
            {segment.match_count} match{segment.match_count !== 1 ? 'es' : ''}
          </span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{
                backgroundColor: `hsl(${segment.peak_similarity * 120}, 70%, 50%)`,
              }}
            />
            {(segment.peak_similarity * 100).toFixed(0)}% similarity
          </span>
        </div>
      )}
    </div>
  );
}
