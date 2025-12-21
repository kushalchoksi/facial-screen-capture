"""
Face detection wrapper for the clip editor.
Provides async-compatible detection with progress callbacks.
"""

import av
import cv2
import numpy as np
from pathlib import Path
import platform
from dataclasses import dataclass
from typing import Callable, Optional
import threading

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("insightface not installed. Run: uv add insightface onnxruntime")


@dataclass
class DetectedSegment:
    """A detected segment where the target face appears."""
    id: str
    start_time: float
    end_time: float
    match_count: int
    peak_similarity: float
    selected: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "match_count": self.match_count,
            "peak_similarity": self.peak_similarity,
            "selected": self.selected,
        }


@dataclass
class DetectionProgress:
    """Progress information during detection."""
    phase: str  # "loading", "pass1", "pass2", "complete"
    progress: float  # 0.0 to 1.0
    message: str


class FaceDetector:
    """Face detector that can run detection with progress updates."""

    def __init__(self):
        self.app: Optional[FaceAnalysis] = None
        self.reference_embeddings: list[np.ndarray] = []
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize the face detection model."""
        if self.app is not None:
            return

        # Try providers in order of preference based on platform
        if platform.system() == 'Darwin':
            providers_to_try = [
                (['CoreMLExecutionProvider', 'CPUExecutionProvider'], 'CoreML'),
                (['CPUExecutionProvider'], 'CPU'),
            ]
        else:
            providers_to_try = [
                (['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA'),
                (['CPUExecutionProvider'], 'CPU'),
            ]

        for providers, name in providers_to_try:
            try:
                self.app = FaceAnalysis(providers=providers)
                ctx_id = 0 if 'CUDA' in name or 'CoreML' in name else -1
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                print(f"Face detection initialized with {name}")
                return
            except Exception as e:
                print(f"{name} not available: {e}")
                continue

        raise RuntimeError("No execution provider available for face detection")

    def load_reference_images(self, folder_path: str) -> int:
        """Load reference face images. Returns count of faces loaded."""
        self.initialize()
        self.reference_embeddings = []

        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Reference folder not found: {folder_path}")

        for image_file in folder.glob('*'):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(image_file))
                if img is None:
                    continue

                faces = self.app.get(img)
                if faces:
                    largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    self.reference_embeddings.append(largest_face.embedding)

        return len(self.reference_embeddings)

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(embedding1, embedding2) /
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def detect_segments(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[DetectionProgress], None]] = None,
        clip_padding: float = 2.0,
        pass1_threshold: float = 0.28,
        pass2_threshold: float = 0.4,
    ) -> list[DetectedSegment]:
        """
        Run two-pass face detection on video.
        Returns list of segments where the target face appears.
        """
        self.initialize()

        if not self.reference_embeddings:
            raise ValueError("No reference faces loaded. Call load_reference_images first.")

        def update_progress(phase: str, progress: float, message: str):
            if progress_callback:
                progress_callback(DetectionProgress(phase, progress, message))

        update_progress("pass1", 0.0, "Starting keyframe scan...")

        # Get video info
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0
        container.close()

        # Pass 1: Keyframe scan
        hot_timestamps = self._pass1_scan(
            video_path, pass1_threshold,
            lambda p, m: update_progress("pass1", p * 0.4, m)  # 0-40%
        )

        if not hot_timestamps:
            update_progress("complete", 1.0, "No matches found")
            return []

        update_progress("pass2", 0.4, "Starting detailed scan...")

        # Pass 2: Detailed scan around hot timestamps
        match_times = self._pass2_scan(
            video_path, hot_timestamps, pass2_threshold,
            lambda p, m: update_progress("pass2", 0.4 + p * 0.5, m)  # 40-90%
        )

        update_progress("complete", 0.95, "Building segments...")

        # Convert match times to segments with padding
        segments = self._build_segments(match_times, clip_padding, duration)

        update_progress("complete", 1.0, f"Found {len(segments)} segments")

        return segments

    def _pass1_scan(
        self,
        video_path: str,
        threshold: float,
        progress_callback: Callable[[float, str], None]
    ) -> list[float]:
        """Pass 1: Quick scan of keyframes."""
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        stream.thread_type = "AUTO"
        time_base = stream.time_base

        # Collect keyframe timestamps
        keyframe_times = []
        for frame in container.decode(stream):
            timestamp = float(frame.pts * time_base) if frame.pts else 0.0
            keyframe_times.append(timestamp)
        container.close()

        if not keyframe_times:
            return []

        # Scan keyframes for faces
        hot_timestamps = []
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for i, target_time in enumerate(keyframe_times):
            progress = i / len(keyframe_times)
            progress_callback(progress, f"Scanning keyframe {i+1}/{len(keyframe_times)}")

            seek_pts = int(target_time / time_base)
            try:
                container.seek(seek_pts, stream=stream)
                frame = next(container.decode(stream))
            except (StopIteration, av.AVError):
                continue

            actual_time = float(frame.pts * time_base) if frame.pts else target_time

            # Downscale for speed
            img = frame.to_ndarray(format='bgr24')
            img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)

            faces = self.app.get(img)
            for face in faces:
                for ref_embedding in self.reference_embeddings:
                    similarity = self._compute_similarity(face.embedding, ref_embedding)
                    if similarity >= threshold:
                        if not any(abs(t - actual_time) < 0.5 for t in hot_timestamps):
                            hot_timestamps.append(actual_time)
                        break
                break

        container.close()
        return hot_timestamps

    def _pass2_scan(
        self,
        video_path: str,
        hot_timestamps: list[float],
        threshold: float,
        progress_callback: Callable[[float, str], None]
    ) -> list[dict]:
        """Pass 2: Detailed scan around hot timestamps."""
        # Merge nearby timestamps into scan regions
        sorted_times = sorted(hot_timestamps)
        regions = []
        current_start = sorted_times[0] - 8.0
        current_end = sorted_times[0] + 8.0

        for t in sorted_times[1:]:
            if t - 8.0 <= current_end:
                current_end = t + 8.0
            else:
                regions.append((max(0, current_start), current_end))
                current_start = t - 8.0
                current_end = t + 8.0
        regions.append((max(0, current_start), current_end))

        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        frame_skip = max(1, int(fps * 0.3))  # Sample every 0.3 seconds

        match_times = []
        total_region_duration = sum(end - start for start, end in regions)
        processed_duration = 0

        for region_idx, (region_start, region_end) in enumerate(regions):
            seek_pts = int(region_start / time_base)
            try:
                container.seek(seek_pts, stream=stream)
            except av.AVError:
                container.seek(0)

            frame_count = 0
            for frame in container.decode(stream):
                current_time = float(frame.pts * time_base) if frame.pts else 0.0

                if current_time > region_end:
                    break
                if current_time < region_start - 0.5:
                    continue

                frame_count += 1
                if frame_skip > 1 and frame_count % frame_skip != 0:
                    continue

                # Update progress
                region_progress = (current_time - region_start) / (region_end - region_start)
                total_progress = (processed_duration + region_progress * (region_end - region_start)) / total_region_duration
                progress_callback(total_progress, f"Scanning region {region_idx+1}/{len(regions)}")

                img = frame.to_ndarray(format='bgr24')
                faces = self.app.get(img)

                for face in faces:
                    for ref_embedding in self.reference_embeddings:
                        similarity = self._compute_similarity(face.embedding, ref_embedding)
                        if similarity >= threshold:
                            match_times.append({
                                'time': current_time,
                                'similarity': similarity
                            })
                            break
                    break

            processed_duration += region_end - region_start

        container.close()
        return match_times

    def _build_segments(
        self,
        match_times: list[dict],
        clip_padding: float,
        video_duration: float
    ) -> list[DetectedSegment]:
        """Convert match timestamps to segments with padding."""
        if not match_times:
            return []

        # Sort by time
        sorted_matches = sorted(match_times, key=lambda x: x['time'])

        # Create initial segments with padding
        raw_segments = []
        for match in sorted_matches:
            start = max(0, match['time'] - clip_padding)
            end = min(video_duration, match['time'] + clip_padding)
            raw_segments.append({
                'start': start,
                'end': end,
                'similarity': match['similarity']
            })

        # Merge overlapping segments
        merged = []
        current = raw_segments[0].copy()
        current['match_count'] = 1
        current['peak_similarity'] = current['similarity']

        for seg in raw_segments[1:]:
            if seg['start'] <= current['end'] + 0.1:
                current['end'] = max(current['end'], seg['end'])
                current['match_count'] += 1
                current['peak_similarity'] = max(current['peak_similarity'], seg['similarity'])
            else:
                merged.append(current)
                current = seg.copy()
                current['match_count'] = 1
                current['peak_similarity'] = current['similarity']

        merged.append(current)

        # Convert to DetectedSegment objects
        return [
            DetectedSegment(
                id=f"seg_{i+1:03d}",
                start_time=seg['start'],
                end_time=seg['end'],
                match_count=seg['match_count'],
                peak_similarity=seg['peak_similarity'],
            )
            for i, seg in enumerate(merged)
        ]


# Global detector instance
_detector: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    """Get or create the global face detector."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector
