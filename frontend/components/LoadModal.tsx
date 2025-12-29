'use client';

import { useState, useRef } from 'react';
import { X, Upload, FolderOpen, Loader2 } from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';

type InputMode = 'file' | 'path';

export function LoadModal() {
  const { isLoadModalOpen, setLoadModalOpen, loadVideo, videoLoaded } =
    useEditorStore();

  const [videoInputMode, setVideoInputMode] = useState<InputMode>('file');
  const [refInputMode, setRefInputMode] = useState<InputMode>('file');
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPath, setVideoPath] = useState('');
  const [refFiles, setRefFiles] = useState<File[]>([]);
  const [refFolder, setRefFolder] = useState('');
  const [clipPadding, setClipPadding] = useState(2.0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const videoInputRef = useRef<HTMLInputElement>(null);
  const refInputRef = useRef<HTMLInputElement>(null);

  if (!isLoadModalOpen) return null;

  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setVideoFile(files[0]);
      setError(null);
    }
  };

  const handleRefFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setRefFiles(Array.from(files));
      setError(null);
    }
  };

  const handleSubmit = async () => {
    setError(null);

    // Validate inputs
    const hasVideo = videoInputMode === 'file' ? videoFile : videoPath.trim();
    const hasRefs = refInputMode === 'file' ? refFiles.length > 0 : refFolder.trim();

    if (!hasVideo) {
      setError('Please select or enter a video');
      return;
    }

    if (!hasRefs) {
      setError('Please select or enter reference images');
      return;
    }

    setIsLoading(true);
    try {
      await loadVideo(
        videoInputMode === 'file' ? videoFile : null,
        videoInputMode === 'path' ? videoPath : null,
        refInputMode === 'file' ? refFiles : null,
        refInputMode === 'path' ? refFolder : null,
        clipPadding
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load video');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    if (!isLoading) {
      setLoadModalOpen(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-dracula-bg rounded-lg shadow-2xl w-full max-w-lg border border-dracula-bg-light">
        <div className="flex items-center justify-between p-4 border-b border-dracula-bg-light">
          <h2 className="text-lg font-semibold">Load Video & References</h2>
          <button
            onClick={handleClose}
            disabled={isLoading}
            className="p-1 rounded hover:bg-dracula-bg-light transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-6">
          {/* Video Input */}
          <div>
            <label className="block text-sm font-medium mb-2">Video Source</label>
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setVideoInputMode('file')}
                className={`flex-1 py-1.5 text-sm rounded transition-colors ${
                  videoInputMode === 'file'
                    ? 'bg-dracula-purple text-white'
                    : 'bg-dracula-bg-light hover:bg-dracula-comment'
                }`}
              >
                Upload File
              </button>
              <button
                onClick={() => setVideoInputMode('path')}
                className={`flex-1 py-1.5 text-sm rounded transition-colors ${
                  videoInputMode === 'path'
                    ? 'bg-dracula-purple text-white'
                    : 'bg-dracula-bg-light hover:bg-dracula-comment'
                }`}
              >
                Server Path
              </button>
            </div>

            {videoInputMode === 'file' ? (
              <div>
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*,.mp4,.mov,.avi,.mkv"
                  onChange={handleVideoFileChange}
                  className="hidden"
                />
                <button
                  onClick={() => videoInputRef.current?.click()}
                  className="w-full py-3 border-2 border-dashed border-dracula-bg-light rounded-lg hover:border-dracula-comment transition-colors"
                >
                  <div className="flex flex-col items-center text-dracula-comment">
                    <Upload className="w-6 h-6 mb-1" />
                    <span className="text-sm">
                      {videoFile ? videoFile.name : 'Click to select video'}
                    </span>
                  </div>
                </button>
              </div>
            ) : (
              <input
                type="text"
                value={videoPath}
                onChange={(e) => setVideoPath(e.target.value)}
                placeholder="/path/to/video.mp4"
                className="w-full px-3 py-2 bg-dracula-bg-dark border border-dracula-bg-light rounded focus:border-dracula-purple focus:outline-none"
              />
            )}
          </div>

          {/* Reference Images Input */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Reference Images
            </label>
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setRefInputMode('file')}
                className={`flex-1 py-1.5 text-sm rounded transition-colors ${
                  refInputMode === 'file'
                    ? 'bg-dracula-purple text-white'
                    : 'bg-dracula-bg-light hover:bg-dracula-comment'
                }`}
              >
                Upload Files
              </button>
              <button
                onClick={() => setRefInputMode('path')}
                className={`flex-1 py-1.5 text-sm rounded transition-colors ${
                  refInputMode === 'path'
                    ? 'bg-dracula-purple text-white'
                    : 'bg-dracula-bg-light hover:bg-dracula-comment'
                }`}
              >
                Server Path
              </button>
            </div>

            {refInputMode === 'file' ? (
              <div>
                <input
                  ref={refInputRef}
                  type="file"
                  accept="image/*,.jpg,.jpeg,.png,.webp"
                  multiple
                  onChange={handleRefFilesChange}
                  className="hidden"
                />
                <button
                  onClick={() => refInputRef.current?.click()}
                  className="w-full py-3 border-2 border-dashed border-dracula-bg-light rounded-lg hover:border-dracula-comment transition-colors"
                >
                  <div className="flex flex-col items-center text-dracula-comment">
                    <FolderOpen className="w-6 h-6 mb-1" />
                    <span className="text-sm">
                      {refFiles.length > 0
                        ? `${refFiles.length} file(s) selected`
                        : 'Click to select images'}
                    </span>
                  </div>
                </button>
              </div>
            ) : (
              <input
                type="text"
                value={refFolder}
                onChange={(e) => setRefFolder(e.target.value)}
                placeholder="/path/to/reference/images/"
                className="w-full px-3 py-2 bg-dracula-bg-dark border border-dracula-bg-light rounded focus:border-dracula-purple focus:outline-none"
              />
            )}
          </div>

          {/* Clip Padding */}
          <div>
            <label className="block text-sm font-medium mb-2">
              Clip Padding: {clipPadding.toFixed(1)}s
            </label>
            <input
              type="range"
              min="0"
              max="10"
              step="0.5"
              value={clipPadding}
              onChange={(e) => setClipPadding(parseFloat(e.target.value))}
              className="w-full accent-dracula-purple"
            />
            <p className="text-xs text-dracula-comment mt-1">
              Extra seconds to add before and after each detected clip
            </p>
          </div>

          {/* Error message */}
          {error && (
            <div className="p-3 bg-dracula-red/20 border border-dracula-red/50 rounded text-sm text-dracula-red">
              {error}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 p-4 border-t border-dracula-bg-light">
          <button
            onClick={handleClose}
            disabled={isLoading}
            className="px-4 py-2 text-sm rounded bg-dracula-bg-light hover:bg-dracula-comment disabled:opacity-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 text-sm rounded bg-dracula-purple hover:bg-dracula-pink disabled:opacity-50 transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Loading...
              </>
            ) : (
              'Load & Detect'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
