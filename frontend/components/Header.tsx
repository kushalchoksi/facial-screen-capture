'use client';

import { Upload, FolderOpen, Download, Film, Loader2 } from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';

export function Header() {
  const {
    videoLoaded,
    isDetecting,
    isExporting,
    detectionStatus,
    segments,
    setLoadModalOpen,
    exportSelectedClips,
  } = useEditorStore();

  const selectedCount = segments.filter((s) => s.selected).length;

  const handleExport = async () => {
    try {
      const exported = await exportSelectedClips();
      alert(`Successfully exported ${exported.length} clips!`);
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Export failed');
    }
  };

  return (
    <header className="flex items-center justify-between px-4 py-3 bg-dracula-bg-dark border-b border-dracula-bg-light">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-dracula-purple">
          <Film className="w-6 h-6" />
          <span className="font-semibold text-lg">Clip Editor</span>
        </div>

        <div className="h-6 w-px bg-dracula-bg-light" />

        <button
          onClick={() => setLoadModalOpen(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded bg-dracula-bg-light hover:bg-dracula-comment transition-colors text-sm"
        >
          <Upload className="w-4 h-4" />
          Load Video
        </button>

        {videoLoaded && (
          <button
            onClick={() => setLoadModalOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded bg-dracula-bg-light hover:bg-dracula-comment transition-colors text-sm"
          >
            <FolderOpen className="w-4 h-4" />
            Load References
          </button>
        )}
      </div>

      {isDetecting && (
        <div className="flex items-center gap-3">
          <Loader2 className="w-4 h-4 animate-spin text-dracula-purple" />
          <div className="text-sm text-dracula-comment">
            {detectionStatus.message || 'Detecting faces...'}
          </div>
          <div className="w-32 h-1.5 bg-dracula-bg-light rounded-full overflow-hidden">
            <div
              className="h-full bg-dracula-purple transition-all duration-300"
              style={{ width: `${detectionStatus.progress * 100}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex items-center gap-3">
        {videoLoaded && segments.length > 0 && (
          <span className="text-sm text-dracula-comment">
            {selectedCount} of {segments.length} clips selected
          </span>
        )}

        <button
          onClick={handleExport}
          disabled={!videoLoaded || selectedCount === 0 || isExporting}
          className="flex items-center gap-2 px-4 py-1.5 rounded bg-dracula-purple hover:bg-dracula-pink disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-medium"
        >
          {isExporting ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          Export Clips
        </button>
      </div>
    </header>
  );
}
