/**
 * Clip Editor - Frontend Application
 * Handles video playback, timeline visualization, and clip management.
 */

// --- State ---
const state = {
    videoLoaded: false,
    duration: 0,
    fps: 24,
    segments: [],
    selectedSegmentId: null,
    dragging: null, // { segmentId, edge: 'start' | 'end' }
    isPlaying: false,
};

// --- DOM Elements ---
const elements = {
    // Modal
    loadModal: document.getElementById('load-modal'),
    videoPathInput: document.getElementById('video-path'),
    referencePathInput: document.getElementById('reference-path'),
    clipPaddingInput: document.getElementById('clip-padding'),
    btnLoad: document.getElementById('btn-load'),
    btnCancelLoad: document.getElementById('btn-cancel-load'),
    btnConfirmLoad: document.getElementById('btn-confirm-load'),
    
    // Tabs & New Inputs
    tabFile: document.getElementById('tab-file'),
    tabPath: document.getElementById('tab-path'),
    inputFileContainer: document.getElementById('input-file-container'),
    inputPathContainer: document.getElementById('input-path-container'),
    videoFileInput: document.getElementById('video-file-input'),

    tabRefFolder: document.getElementById('tab-ref-folder'),
    tabRefPath: document.getElementById('tab-ref-path'),
    inputRefFolderContainer: document.getElementById('input-ref-folder-container'),
    inputRefPathContainer: document.getElementById('input-ref-path-container'),
    referenceFolderInput: document.getElementById('reference-folder-input'),

    // Video
    video: document.getElementById('video-player'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    btnPlay: document.getElementById('btn-play'),
    iconPlay: document.getElementById('icon-play'),
    iconPause: document.getElementById('icon-pause'),
    btnPrevFrame: document.getElementById('btn-prev-frame'),
    btnNextFrame: document.getElementById('btn-next-frame'),
    timeCurrent: document.getElementById('time-current'),
    timeDuration: document.getElementById('time-duration'),
    fpsDisplay: document.getElementById('fps-display'),

    // Timeline
    timelineCanvas: document.getElementById('timeline-canvas'),
    playhead: document.getElementById('playhead'),

    // Progress
    progressContainer: document.getElementById('progress-container'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),

    // Clips
    clipsList: document.getElementById('clips-list'),
    btnSelectAll: document.getElementById('btn-select-all'),
    btnDeselectAll: document.getElementById('btn-deselect-all'),
    btnAddClip: document.getElementById('btn-add-clip'),
    btnExport: document.getElementById('btn-export'),

    // Status
    statusText: document.getElementById('status-text'),
};

// --- Utilities ---

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function formatTimePrecise(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = (seconds % 60).toFixed(1);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.padStart(4, '0')}`;
}

async function api(method, endpoint, body = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    const response = await fetch(endpoint, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
}

// --- Modal ---

function showLoadModal() {
    elements.loadModal.classList.remove('hidden');
    // Focus appropriate input based on active tab
    if (elements.tabPath.classList.contains('active')) {
        elements.videoPathInput.focus();
    }
}

function hideLoadModal() {
    elements.loadModal.classList.add('hidden');
}

// --- Tabs ---

function switchTab(tab) {
    if (tab === 'file') {
        elements.tabFile.classList.add('active');
        elements.tabPath.classList.remove('active');
        elements.inputFileContainer.classList.remove('hidden');
        elements.inputPathContainer.classList.add('hidden');
    } else {
        elements.tabFile.classList.remove('active');
        elements.tabPath.classList.add('active');
        elements.inputFileContainer.classList.add('hidden');
        elements.inputPathContainer.classList.remove('hidden');
        setTimeout(() => elements.videoPathInput.focus(), 50);
    }
}

function switchRefTab(tab) {
    if (tab === 'folder') {
        elements.tabRefFolder.classList.add('active');
        elements.tabRefPath.classList.remove('active');
        elements.inputRefFolderContainer.classList.remove('hidden');
        elements.inputRefPathContainer.classList.add('hidden');
    } else {
        elements.tabRefFolder.classList.remove('active');
        elements.tabRefPath.classList.add('active');
        elements.inputRefFolderContainer.classList.add('hidden');
        elements.inputRefPathContainer.classList.remove('hidden');
        setTimeout(() => elements.referencePathInput.focus(), 50);
    }
}

elements.tabFile.addEventListener('click', () => switchTab('file'));
elements.tabPath.addEventListener('click', () => switchTab('path'));
elements.tabRefFolder.addEventListener('click', () => switchRefTab('folder'));
elements.tabRefPath.addEventListener('click', () => switchRefTab('path'));


// --- Video Loading ---

async function loadVideo() {
    const isVideoFileTab = elements.tabFile.classList.contains('active');
    const isRefFolderTab = elements.tabRefFolder.classList.contains('active');
    const clipPadding = parseFloat(elements.clipPaddingInput.value) || 2.0;

    // Validate inputs
    let videoFile, videoPath, refFiles, refPath;

    if (isVideoFileTab) {
        videoFile = elements.videoFileInput.files[0];
        if (!videoFile) return alert('Please select a video file');
    } else {
        videoPath = elements.videoPathInput.value.trim();
        if (!videoPath) return alert('Please provide video path');
    }

    if (isRefFolderTab) {
        refFiles = elements.referenceFolderInput.files;
        if (!refFiles || refFiles.length === 0) return alert('Please select a reference folder with images');
    } else {
        refPath = elements.referencePathInput.value.trim();
        if (!refPath) return alert('Please provide reference folder path');
    }

    hideLoadModal();
    showProgress();

    try {
        let result;

        // If EITHER is a file upload, we must use the multipart endpoint
        if (isVideoFileTab || isRefFolderTab) {
            elements.progressText.textContent = 'Uploading data...';
            const formData = new FormData();
            
            formData.append('clip_padding', clipPadding);

            if (isVideoFileTab) {
                formData.append('video_file', videoFile);
            } else {
                formData.append('video_path', videoPath);
            }

            if (isRefFolderTab) {
                let addedCount = 0;
                for (let i = 0; i < refFiles.length; i++) {
                    const file = refFiles[i];
                    // Only append images
                    // Check MIME type or extension
                    const isImage = file.type.startsWith('image/') || 
                                  /\.(jpg|jpeg|png|bmp|webp|tiff|gif)$/i.test(file.name);
                    
                    if (isImage) {
                        formData.append('reference_files', file);
                        addedCount++;
                    }
                }
                
                if (addedCount === 0) {
                    hideProgress();
                    alert('No valid images (jpg, png, etc.) found in the selected folder.');
                    return;
                }
                console.log(`Uploading ${addedCount} reference images`);
            } else {
                formData.append('reference_folder', refPath);
            }

            const response = await fetch('/api/load_upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            result = await response.json();

        } else {
            // Both are paths - use JSON endpoint
            result = await api('POST', '/api/load', {
                video_path: videoPath,
                reference_folder: refPath,
                clip_padding: clipPadding,
            });
        }

        // Update state
        state.duration = result.video_info.duration;
        state.fps = result.video_info.fps;
        state.videoLoaded = true;

        // Update UI
        elements.video.src = '/video/stream';
        elements.videoPlaceholder.classList.add('hidden');
        elements.timeDuration.textContent = formatTime(state.duration);
        elements.fpsDisplay.textContent = `${state.fps.toFixed(1)} fps`;
        elements.btnAddClip.disabled = false;

        // Resize timeline
        resizeTimeline();

        // Start polling for detection progress
        pollDetectionProgress();

    } catch (error) {
        hideProgress();
        alert(`Failed to load: ${error.message}`);
    }
}

// --- Detection Progress ---

function showProgress() {
    elements.progressContainer.classList.remove('hidden');
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = 'Starting...';
}

function hideProgress() {
    elements.progressContainer.classList.add('hidden');
}

async function pollDetectionProgress() {
    try {
        const status = await api('GET', '/api/status');

        elements.progressFill.style.width = `${status.progress * 100}%`;
        elements.progressText.textContent = status.message;
        elements.statusText.textContent = status.message;

        if (status.running) {
            setTimeout(pollDetectionProgress, 500);
        } else {
            hideProgress();
            // Fetch segments
            await loadSegments();
        }
    } catch (error) {
        console.error('Progress poll error:', error);
        setTimeout(pollDetectionProgress, 1000);
    }
}

// --- Segments ---

async function loadSegments() {
    try {
        const result = await api('GET', '/api/segments');
        state.segments = result.segments;
        renderClipsList();
        renderTimeline();
        elements.btnExport.disabled = state.segments.length === 0;
    } catch (error) {
        console.error('Failed to load segments:', error);
    }
}

function renderClipsList() {
    if (state.segments.length === 0) {
        elements.clipsList.innerHTML = '<p class="clips-empty">No clips detected</p>';
        return;
    }

    elements.clipsList.innerHTML = state.segments.map((seg, index) => `
        <div class="clip-item ${seg.id === state.selectedSegmentId ? 'active' : ''}" data-id="${seg.id}">
            <div class="clip-item-header">
                <input type="checkbox" ${seg.selected ? 'checked' : ''} data-action="toggle-select">
                <span class="clip-name">Clip ${index + 1}</span>
            </div>
            <div class="clip-item-time">
                ${formatTimePrecise(seg.start_time)} - ${formatTimePrecise(seg.end_time)}
            </div>
            <div class="clip-item-meta">
                <span>Duration: ${(seg.end_time - seg.start_time).toFixed(1)}s</span>
                <span>Similarity: ${(seg.peak_similarity * 100).toFixed(0)}%</span>
            </div>
            <div class="clip-item-actions">
                <button class="btn btn-sm" data-action="preview">Preview</button>
                <button class="btn btn-sm btn-danger" data-action="delete">Delete</button>
            </div>
        </div>
    `).join('');
}

// --- Clip Actions ---

elements.clipsList.addEventListener('click', async (e) => {
    const item = e.target.closest('.clip-item');
    if (!item) return;

    const segmentId = item.dataset.id;
    const action = e.target.dataset.action;

    if (action === 'toggle-select') {
        await toggleSegmentSelection(segmentId, e.target.checked);
    } else if (action === 'preview') {
        previewSegment(segmentId);
    } else if (action === 'delete') {
        await deleteSegment(segmentId);
    } else {
        // Click on clip item - select it
        selectSegment(segmentId);
    }
});

function selectSegment(segmentId) {
    state.selectedSegmentId = segmentId;
    renderClipsList();

    // Seek to segment start
    const segment = state.segments.find(s => s.id === segmentId);
    if (segment) {
        elements.video.currentTime = segment.start_time;
    }
}

function previewSegment(segmentId) {
    const segment = state.segments.find(s => s.id === segmentId);
    if (segment) {
        elements.video.currentTime = segment.start_time;
        elements.video.play();
    }
}

async function toggleSegmentSelection(segmentId, selected) {
    try {
        await api('PUT', `/api/segments/${segmentId}`, { selected });
        const segment = state.segments.find(s => s.id === segmentId);
        if (segment) {
            segment.selected = selected;
        }
    } catch (error) {
        console.error('Failed to update segment:', error);
    }
}

async function deleteSegment(segmentId) {
    if (!confirm('Delete this clip?')) return;

    try {
        await api('DELETE', `/api/segments/${segmentId}`);
        state.segments = state.segments.filter(s => s.id !== segmentId);
        if (state.selectedSegmentId === segmentId) {
            state.selectedSegmentId = null;
        }
        renderClipsList();
        renderTimeline();
    } catch (error) {
        console.error('Failed to delete segment:', error);
    }
}

elements.btnSelectAll.addEventListener('click', async () => {
    for (const seg of state.segments) {
        if (!seg.selected) {
            await toggleSegmentSelection(seg.id, true);
        }
    }
    renderClipsList();
});

elements.btnDeselectAll.addEventListener('click', async () => {
    for (const seg of state.segments) {
        if (seg.selected) {
            await toggleSegmentSelection(seg.id, false);
        }
    }
    renderClipsList();
});

// --- Export ---

elements.btnExport.addEventListener('click', async () => {
    const selectedIds = state.segments.filter(s => s.selected).map(s => s.id);

    if (selectedIds.length === 0) {
        alert('No clips selected for export');
        return;
    }

    elements.btnExport.disabled = true;
    elements.btnExport.textContent = 'Exporting...';

    try {
        const result = await api('POST', '/api/export', { segment_ids: selectedIds });
        alert(`Exported ${result.exported.length} clips to:\n${result.output_folder}`);
    } catch (error) {
        alert(`Export failed: ${error.message}`);
    } finally {
        elements.btnExport.disabled = false;
        elements.btnExport.textContent = 'Export Selected';
    }
});

// --- Timeline ---

function resizeTimeline() {
    const canvas = elements.timelineCanvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    renderTimeline();
}

function renderTimeline() {
    const canvas = elements.timelineCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const dpr = window.devicePixelRatio;

    // Clear
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, width, height);

    if (!state.videoLoaded || state.duration === 0) return;

    // Draw time markers
    ctx.fillStyle = '#444';
    ctx.font = `${10 * dpr}px -apple-system, sans-serif`;

    const interval = getTimeInterval(state.duration);
    for (let t = 0; t <= state.duration; t += interval) {
        const x = (t / state.duration) * width;
        ctx.fillRect(x, 0, 1 * dpr, 8 * dpr);
        if (t > 0 && t < state.duration) {
            ctx.fillStyle = '#666';
            ctx.fillText(formatTime(t), x + 4 * dpr, 20 * dpr);
            ctx.fillStyle = '#444';
        }
    }

    // Draw segments
    const segmentY = 30 * dpr;
    const segmentHeight = 24 * dpr;

    for (const seg of state.segments) {
        const startX = (seg.start_time / state.duration) * width;
        const endX = (seg.end_time / state.duration) * width;
        const segWidth = Math.max(endX - startX, 4 * dpr);

        // Segment fill
        ctx.fillStyle = seg.id === state.selectedSegmentId ? '#6bb3ff' : '#4a9eff';
        ctx.globalAlpha = seg.selected ? 0.8 : 0.4;
        ctx.fillRect(startX, segmentY, segWidth, segmentHeight);
        ctx.globalAlpha = 1;

        // Segment border
        ctx.strokeStyle = seg.id === state.selectedSegmentId ? '#fff' : '#6bb3ff';
        ctx.lineWidth = seg.id === state.selectedSegmentId ? 2 * dpr : 1 * dpr;
        ctx.strokeRect(startX, segmentY, segWidth, segmentHeight);

        // Drag handles (only for selected)
        if (seg.id === state.selectedSegmentId) {
            ctx.fillStyle = '#fff';
            ctx.fillRect(startX - 2 * dpr, segmentY, 4 * dpr, segmentHeight);
            ctx.fillRect(startX + segWidth - 2 * dpr, segmentY, 4 * dpr, segmentHeight);
        }
    }
}

function getTimeInterval(duration) {
    if (duration < 60) return 5;
    if (duration < 300) return 30;
    if (duration < 600) return 60;
    if (duration < 1800) return 120;
    return 300;
}

function updatePlayhead() {
    if (!state.videoLoaded) return;

    const progress = elements.video.currentTime / state.duration;
    const container = elements.timelineCanvas.parentElement;
    elements.playhead.style.left = `${progress * container.offsetWidth}px`;
    elements.timeCurrent.textContent = formatTime(elements.video.currentTime);
}

// Timeline click to seek
elements.timelineCanvas.addEventListener('click', (e) => {
    if (!state.videoLoaded) return;

    const rect = elements.timelineCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const progress = x / rect.width;
    elements.video.currentTime = progress * state.duration;
});

// Timeline drag for segment adjustment
elements.timelineCanvas.addEventListener('mousedown', (e) => {
    if (!state.selectedSegmentId || !state.videoLoaded) return;

    const rect = elements.timelineCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / rect.width) * state.duration;

    const segment = state.segments.find(s => s.id === state.selectedSegmentId);
    if (!segment) return;

    const handleSize = 10; // pixels
    const startX = (segment.start_time / state.duration) * rect.width;
    const endX = (segment.end_time / state.duration) * rect.width;

    if (Math.abs(x - startX) < handleSize) {
        state.dragging = { segmentId: segment.id, edge: 'start' };
    } else if (Math.abs(x - endX) < handleSize) {
        state.dragging = { segmentId: segment.id, edge: 'end' };
    }
});

document.addEventListener('mousemove', (e) => {
    if (!state.dragging) return;

    const rect = elements.timelineCanvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const time = (x / rect.width) * state.duration;

    const segment = state.segments.find(s => s.id === state.dragging.segmentId);
    if (!segment) return;

    if (state.dragging.edge === 'start') {
        segment.start_time = Math.max(0, Math.min(time, segment.end_time - 0.5));
    } else {
        segment.end_time = Math.max(segment.start_time + 0.5, Math.min(time, state.duration));
    }

    renderTimeline();
    renderClipsList();
});

document.addEventListener('mouseup', async () => {
    if (!state.dragging) return;

    const segment = state.segments.find(s => s.id === state.dragging.segmentId);
    if (segment) {
        try {
            await api('PUT', `/api/segments/${segment.id}`, {
                start_time: segment.start_time,
                end_time: segment.end_time,
            });
        } catch (error) {
            console.error('Failed to update segment:', error);
        }
    }

    state.dragging = null;
});

// --- Video Controls ---

elements.btnPlay.addEventListener('click', togglePlay);

function togglePlay() {
    if (elements.video.paused) {
        elements.video.play();
    } else {
        elements.video.pause();
    }
}

elements.video.addEventListener('play', () => {
    state.isPlaying = true;
    elements.iconPlay.classList.add('hidden');
    elements.iconPause.classList.remove('hidden');
});

elements.video.addEventListener('pause', () => {
    state.isPlaying = false;
    elements.iconPlay.classList.remove('hidden');
    elements.iconPause.classList.add('hidden');
});

elements.video.addEventListener('timeupdate', updatePlayhead);
elements.video.addEventListener('loadedmetadata', () => {
    state.duration = elements.video.duration;
    elements.timeDuration.textContent = formatTime(state.duration);
    renderTimeline();
});

// Frame stepping
function stepFrame(direction) {
    elements.video.pause();
    const frameTime = 1 / state.fps;
    elements.video.currentTime = Math.max(0, Math.min(
        elements.video.currentTime + direction * frameTime,
        state.duration
    ));
}

elements.btnPrevFrame.addEventListener('click', () => stepFrame(-1));
elements.btnNextFrame.addEventListener('click', () => stepFrame(1));

// --- Keyboard Shortcuts ---

document.addEventListener('keydown', (e) => {
    // Ignore if typing in input
    if (e.target.tagName === 'INPUT') return;

    switch (e.key) {
        case ' ':
            e.preventDefault();
            togglePlay();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            stepFrame(-1);
            break;
        case 'ArrowRight':
            e.preventDefault();
            stepFrame(1);
            break;
        case 'j':
            elements.video.currentTime = Math.max(0, elements.video.currentTime - 5);
            break;
        case 'l':
            elements.video.currentTime = Math.min(state.duration, elements.video.currentTime + 5);
            break;
        case 'Home':
            elements.video.currentTime = 0;
            break;
        case 'End':
            elements.video.currentTime = state.duration;
            break;
    }
});

// --- Window Events ---

window.addEventListener('resize', () => {
    if (state.videoLoaded) {
        resizeTimeline();
    }
});

// --- Modal Events ---

elements.btnLoad.addEventListener('click', showLoadModal);
elements.btnCancelLoad.addEventListener('click', hideLoadModal);
elements.btnConfirmLoad.addEventListener('click', loadVideo);

elements.loadModal.addEventListener('click', (e) => {
    if (e.target === elements.loadModal) {
        hideLoadModal();
    }
});

// Enter key in modal inputs
elements.clipPaddingInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadVideo();
});

// --- Add Manual Clip ---

elements.btnAddClip.addEventListener('click', async () => {
    const currentTime = elements.video.currentTime;
    const startTime = Math.max(0, currentTime - 2);
    const endTime = Math.min(state.duration, currentTime + 2);

    try {
        const result = await api('POST', '/api/segments', {
            start_time: startTime,
            end_time: endTime,
        });
        state.segments.push(result);
        state.segments.sort((a, b) => a.start_time - b.start_time);
        state.selectedSegmentId = result.id;
        renderClipsList();
        renderTimeline();
        elements.btnExport.disabled = false;
    } catch (error) {
        console.error('Failed to add segment:', error);
        alert(`Failed to add clip: ${error.message}`);
    }
});

// --- Initialize ---

console.log('Clip Editor initialized');
