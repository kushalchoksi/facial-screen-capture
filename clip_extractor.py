"""
Clip Extractor - Extract video clips containing a specific person's face.

Uses the same two-pass face detection strategy as main.py, but outputs
video clips instead of individual frames.
"""

import av
import cv2
import numpy as np
from fractions import Fraction
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Optional
import traceback

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: insightface not installed")
    print("Run: uv add insightface onnxruntime")
    exit(1)


@dataclass
class ClipConfig:
    """Configuration for clip extraction."""
    # Input/Output
    video_path: str = "test.mp4"
    reference_folder: str = "reference_images"
    output_folder: str = "clips"
    output_relative_to_video: bool = True

    # Pass 1 settings (reconnaissance)
    pass1_scale: float = 0.6
    pass1_similarity_threshold: float = 0.28
    pass1_sample_between_keyframes: int = 2

    # Pass 2 settings (segment detection)
    pass2_sample_interval: float = 0.3
    pass2_segment_padding: float = 8.0
    pass2_similarity_threshold: float = 0.4

    # Clip settings
    clip_padding: float = 2.0  # seconds before/after each segment
    min_clip_duration: float = 1.0  # skip clips shorter than this
    include_audio: bool = True

    # Time budget
    max_processing_time: int = 300

    # Output
    manifest_file: str = "clips_manifest.txt"

    def get_output_path(self) -> Path:
        """Get the resolved output folder path."""
        if self.output_relative_to_video:
            video_dir = Path(self.video_path).parent
            return video_dir / self.output_folder
        return Path(self.output_folder)

    def get_manifest_path(self) -> Path:
        """Get the resolved manifest file path."""
        return self.get_output_path() / self.manifest_file


@dataclass
class ClipSegment:
    """A detected segment where the target face appears."""
    start_time: float
    end_time: float
    start_formatted: str
    end_formatted: str
    match_count: int  # number of face matches in this segment
    peak_similarity: float


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    pass1_keyframes_total: int = 0
    pass1_keyframes_with_faces: int = 0
    pass1_hot_segments: int = 0
    pass1_time: float = 0.0
    pass2_frames_processed: int = 0
    pass2_segments_found: int = 0
    pass2_time: float = 0.0
    clips_extracted: int = 0
    extraction_time: float = 0.0
    total_time: float = 0.0


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_filename(seconds: float) -> str:
    """Convert seconds to HH-MM-SS format suitable for filenames."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(embedding1, embedding2) /
                (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))


def load_reference_faces(sample_folder: str, app: FaceAnalysis) -> list[np.ndarray]:
    """Load and encode reference face images."""
    print(f"Loading reference images from {sample_folder}...")
    reference_embeddings = []

    for image_file in Path(sample_folder).glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"  Processing {image_file.name}")
            img = cv2.imread(str(image_file))

            if img is None:
                print(f"    Warning: Could not read {image_file.name}")
                continue

            faces = app.get(img)

            if faces:
                largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                reference_embeddings.append(largest_face.embedding)
                print(f"    Face detected and encoded")
            else:
                print(f"    Warning: No face found in {image_file.name}")

    print(f"Loaded {len(reference_embeddings)} reference face(s)\n")
    return reference_embeddings


def get_video_info(video_path: str) -> dict:
    """Get video metadata without loading frames."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Check for audio stream
    has_audio = len(container.streams.audio) > 0

    info = {
        'duration': float(container.duration / av.time_base) if container.duration else 0,
        'fps': float(stream.average_rate) if stream.average_rate else 24.0,
        'total_frames': stream.frames if stream.frames else 0,
        'width': stream.width,
        'height': stream.height,
        'codec': stream.codec_context.name,
        'has_audio': has_audio,
    }

    container.close()
    return info


def initialize_face_app() -> FaceAnalysis:
    """Initialize InsightFace with best available provider."""
    import platform
    print("Initializing face detection...")

    # Try providers in order of preference based on platform
    if platform.system() == 'Darwin':
        # macOS: try CoreML first (Apple Silicon), then CPU
        providers_to_try = [
            (['CoreMLExecutionProvider', 'CPUExecutionProvider'], 'CoreML'),
            (['CPUExecutionProvider'], 'CPU'),
        ]
    else:
        # Linux/Windows: try CUDA first, then CPU
        providers_to_try = [
            (['CUDAExecutionProvider', 'CPUExecutionProvider'], 'CUDA'),
            (['CPUExecutionProvider'], 'CPU'),
        ]

    for providers, name in providers_to_try:
        try:
            app = FaceAnalysis(providers=providers)
            ctx_id = 0 if 'CUDA' in name or 'CoreML' in name else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(f"  Using {name}")
            print()
            return app
        except Exception as e:
            print(f"  {name} not available: {e}")
            continue

    raise RuntimeError("No execution provider available for face detection")


def pass1_keyframe_scan(
    video_path: str,
    reference_embeddings: list[np.ndarray],
    app: FaceAnalysis,
    config: ClipConfig,
    start_time: float
) -> tuple[list[dict], ProcessingStats]:
    """
    Pass 1: Fast scan of keyframes + sampled inter-keyframe frames.
    Returns timestamps of segments containing the target face.
    """
    print("=" * 50)
    print("PASS 1: Keyframe Reconnaissance")
    print("=" * 50)

    stats = ProcessingStats()
    hot_segments = []
    pass1_start = time.time()

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"
    stream.thread_type = "AUTO"

    time_base = stream.time_base

    print(f"  Scale factor: {config.pass1_scale}")
    print(f"  Similarity threshold: {config.pass1_similarity_threshold}")
    print(f"  Inter-keyframe samples: {config.pass1_sample_between_keyframes}")
    print()

    # Collect keyframe timestamps first
    keyframe_timestamps = []
    print("  Phase 1a: Mapping keyframes...")

    for frame in container.decode(stream):
        timestamp = float(frame.pts * time_base) if frame.pts else 0.0
        keyframe_timestamps.append(timestamp)

    container.close()
    print(f"  Found {len(keyframe_timestamps)} keyframes")

    # Build sample points: keyframes + samples between them
    sample_points = []
    for i, kf_time in enumerate(keyframe_timestamps):
        sample_points.append(('keyframe', kf_time))

        if i < len(keyframe_timestamps) - 1 and config.pass1_sample_between_keyframes > 0:
            next_kf_time = keyframe_timestamps[i + 1]
            gap = next_kf_time - kf_time

            for j in range(1, config.pass1_sample_between_keyframes + 1):
                inter_time = kf_time + (gap * j / (config.pass1_sample_between_keyframes + 1))
                sample_points.append(('inter', inter_time))

    print(f"  Total sample points: {len(sample_points)}")
    print()

    # Process sample points
    print("  Phase 1b: Scanning for faces...")
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    processed_count = 0
    last_progress_time = time.time()
    last_seek_time = -999

    for sample_type, target_time in sample_points:
        elapsed = time.time() - start_time
        if elapsed > config.max_processing_time * 0.5:
            print(f"\n  Time budget reached for Pass 1 ({elapsed:.1f}s)")
            break

        if time.time() - last_progress_time > 3.0:
            print(f"  Scanning... {format_timestamp(target_time)} "
                  f"({processed_count}/{len(sample_points)} samples, {len(hot_segments)} hits)")
            last_progress_time = time.time()

        if abs(target_time - last_seek_time) > 0.5:
            seek_pts = int(target_time / time_base)
            try:
                container.seek(seek_pts, stream=stream)
            except av.AVError:
                continue

        try:
            frame = next(container.decode(stream))
        except (StopIteration, av.AVError):
            continue

        actual_time = float(frame.pts * time_base) if frame.pts else target_time
        last_seek_time = actual_time
        processed_count += 1

        if sample_type == 'keyframe':
            stats.pass1_keyframes_total += 1

        img = frame.to_ndarray(format='bgr24')

        if config.pass1_scale != 1.0:
            img = cv2.resize(img, None,
                           fx=config.pass1_scale,
                           fy=config.pass1_scale,
                           interpolation=cv2.INTER_LINEAR)

        faces = app.get(img)

        if faces:
            if sample_type == 'keyframe':
                stats.pass1_keyframes_with_faces += 1

            for face in faces:
                for ref_embedding in reference_embeddings:
                    similarity = compute_similarity(face.embedding, ref_embedding)

                    if similarity >= config.pass1_similarity_threshold:
                        if not any(abs(h['timestamp'] - actual_time) < 0.3 for h in hot_segments):
                            hot_segments.append({
                                'pts': frame.pts,
                                'timestamp': actual_time,
                                'similarity': similarity,
                                'source': sample_type
                            })
                            print(f"  [HIT] {format_timestamp(actual_time)} "
                                  f"(similarity: {similarity:.3f}, {sample_type})")
                        break
                break

    container.close()

    stats.pass1_hot_segments = len(hot_segments)
    stats.pass1_time = time.time() - pass1_start

    print()
    print(f"Pass 1 Complete:")
    print(f"  Samples processed: {processed_count}")
    print(f"  Hot segments found: {stats.pass1_hot_segments}")
    print(f"  Time: {stats.pass1_time:.1f}s")
    print()

    return hot_segments, stats


def merge_hot_segments_for_scanning(segments: list[dict], max_gap: float = 15.0) -> list[tuple[float, float]]:
    """
    Merge adjacent hot segments for Pass 2 scanning.
    Returns list of (start_time, end_time) tuples.
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x['timestamp'])

    merged = []
    current_start = sorted_segments[0]['timestamp']
    current_end = sorted_segments[0]['timestamp']

    for seg in sorted_segments[1:]:
        if seg['timestamp'] - current_end <= max_gap:
            current_end = seg['timestamp']
        else:
            merged.append((current_start, current_end))
            current_start = seg['timestamp']
            current_end = seg['timestamp']

    merged.append((current_start, current_end))

    return merged


def pass2_segment_detection(
    video_path: str,
    hot_segments: list[dict],
    reference_embeddings: list[np.ndarray],
    app: FaceAnalysis,
    config: ClipConfig,
    stats: ProcessingStats,
    start_time: float
) -> list[ClipSegment]:
    """
    Pass 2: Detect continuous segments where the target face appears.
    Returns list of ClipSegment with precise start/end times.
    """
    print("=" * 50)
    print("PASS 2: Segment Detection")
    print("=" * 50)

    if not hot_segments:
        print("  No hot segments to process")
        return []

    # Merge for efficient scanning
    scan_regions = merge_hot_segments_for_scanning(hot_segments, max_gap=15.0)
    print(f"  Merged {len(hot_segments)} hits into {len(scan_regions)} scan regions")
    print(f"  Sample interval: {config.pass2_sample_interval}s")
    print(f"  Segment padding: {config.pass2_segment_padding}s")
    print()

    pass2_start = time.time()
    all_match_times = []  # Collect all match timestamps
    frames_processed = 0

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    time_base = stream.time_base
    fps = float(stream.average_rate) if stream.average_rate else 24.0
    frame_skip = max(1, int(fps * config.pass2_sample_interval))

    for region_idx, (region_start, region_end) in enumerate(scan_regions):
        elapsed = time.time() - start_time
        remaining = config.max_processing_time - elapsed
        if remaining < 10:
            print(f"\n  Time budget exhausted")
            break

        actual_start = max(0, region_start - config.pass2_segment_padding)
        actual_end = region_end + config.pass2_segment_padding

        print(f"  Processing region {region_idx + 1}/{len(scan_regions)}: "
              f"{format_timestamp(actual_start)} - {format_timestamp(actual_end)}")

        seek_pts = int(actual_start / time_base)
        try:
            container.seek(seek_pts, stream=stream)
        except av.AVError:
            container.seek(0)

        frame_count = 0
        region_matches = 0

        for frame in container.decode(stream):
            current_time = float(frame.pts * time_base) if frame.pts else 0.0

            if current_time > actual_end:
                break

            if current_time < actual_start - 0.5:
                continue

            frame_count += 1
            if frame_skip > 1 and frame_count % frame_skip != 0:
                continue

            frames_processed += 1

            img = frame.to_ndarray(format='bgr24')
            faces = app.get(img)

            for face in faces:
                for ref_embedding in reference_embeddings:
                    similarity = compute_similarity(face.embedding, ref_embedding)

                    if similarity >= config.pass2_similarity_threshold:
                        all_match_times.append({
                            'time': current_time,
                            'similarity': similarity
                        })
                        region_matches += 1
                        break
                break

        print(f"    -> {region_matches} matches in this region")

    container.close()

    # Convert match times to segments (each match becomes its own segment)
    segments = []
    for match in all_match_times:
        # Apply clip padding
        seg_start = max(0, match['time'] - config.clip_padding)
        seg_end = match['time'] + config.clip_padding

        segments.append(ClipSegment(
            start_time=seg_start,
            end_time=seg_end,
            start_formatted=format_timestamp(seg_start),
            end_formatted=format_timestamp(seg_end),
            match_count=1,
            peak_similarity=match['similarity']
        ))

    # Merge overlapping segments
    segments = merge_overlapping_segments(segments)

    stats.pass2_frames_processed = frames_processed
    stats.pass2_segments_found = len(segments)
    stats.pass2_time = time.time() - pass2_start

    print()
    print(f"Pass 2 Complete:")
    print(f"  Frames processed: {stats.pass2_frames_processed}")
    print(f"  Segments detected: {stats.pass2_segments_found}")
    print(f"  Time: {stats.pass2_time:.1f}s")
    print()

    return segments


def merge_overlapping_segments(segments: list[ClipSegment]) -> list[ClipSegment]:
    """Merge segments that overlap or are adjacent."""
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda x: x.start_time)

    merged = []
    current = sorted_segs[0]

    for seg in sorted_segs[1:]:
        # If overlapping or adjacent (within 0.1s), merge
        if seg.start_time <= current.end_time + 0.1:
            # Extend current segment
            current = ClipSegment(
                start_time=current.start_time,
                end_time=max(current.end_time, seg.end_time),
                start_formatted=current.start_formatted,
                end_formatted=format_timestamp(max(current.end_time, seg.end_time)),
                match_count=current.match_count + seg.match_count,
                peak_similarity=max(current.peak_similarity, seg.peak_similarity)
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)
    return merged


def extract_clip_pyav(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: Path,
    include_audio: bool = True
) -> bool:
    """
    Extract a clip from video using PyAV with stream copy (remuxing).
    Returns True on success, False on failure.
    """
    try:
        with av.open(video_path) as in_container:
            out_container = av.open(str(output_path), mode='w')

            in_streams = []
            out_streams = []
            for in_stream in in_container.streams:
                if in_stream.type in ('video', 'audio'):
                    if not include_audio and in_stream.type == 'audio':
                        continue
                    out_stream = out_container.add_stream_from_template(in_stream, opaque=True)
                    in_streams.append(in_stream)
                    out_streams.append(out_stream)

            if not in_streams:
                out_container.close()
                return False
            
            # Seek to the start time
            in_container.seek(int(start_time * av.time_base), backward=True, any_frame=False, stream=None)

            pts_offset_map = {s: None for s in in_streams}
            
            for packet in in_container.demux(in_streams):
                if packet.dts is None:
                    continue

                packet_time_in_seconds = packet.pts * float(packet.stream.time_base)

                if pts_offset_map[packet.stream] is None:
                    pts_offset_map[packet.stream] = packet.pts

                if packet_time_in_seconds < end_time:
                    if packet.pts is None or pts_offset_map[packet.stream] is None:
                        continue
                    
                    packet.pts -= pts_offset_map[packet.stream]
                    packet.dts -= pts_offset_map[packet.stream]

                    if packet.pts < 0:
                        packet.pts = 0
                    if packet.dts < 0:
                        packet.dts = 0
                    
                    for i, in_s in enumerate(in_streams):
                        if in_s == packet.stream:
                            packet.stream = out_streams[i]
                            break
                    
                    out_container.mux(packet)
                else:
                    if packet.stream.type == 'video':
                        break
            
            out_container.close()
        return True

    except Exception as e:
        traceback.print_exc()
        print(f"    Error extracting clip: {e}")
        return False


def extract_all_clips(
    video_path: str,
    segments: list[ClipSegment],
    config: ClipConfig,
    stats: ProcessingStats
) -> list[Path]:
    """Extract all clips from the video."""
    print("=" * 50)
    print("CLIP EXTRACTION")
    print("=" * 50)

    if not segments:
        print("  No segments to extract")
        return []

    output_folder = config.get_output_path()
    output_folder.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    video_ext = Path(video_path).suffix

    print(f"  Output folder: {output_folder}")
    print(f"  Segments to extract: {len(segments)}")
    print(f"  Include audio: {config.include_audio}")
    print()

    extraction_start = time.time()
    extracted_clips = []

    for i, segment in enumerate(segments):
        # Skip very short clips
        duration = segment.end_time - segment.start_time
        if duration < config.min_clip_duration:
            print(f"  Skipping clip {i+1} (duration {duration:.1f}s < {config.min_clip_duration}s)")
            continue

        # Generate filename
        start_str = format_timestamp_filename(segment.start_time)
        end_str = format_timestamp_filename(segment.end_time)
        clip_filename = f"clip_{i+1:03d}_{start_str}_to_{end_str}{video_ext}"
        clip_path = output_folder / clip_filename

        print(f"  Extracting clip {i+1}/{len(segments)}: "
              f"{segment.start_formatted} - {segment.end_formatted} ({duration:.1f}s)")

        success = extract_clip_pyav(
            video_path,
            segment.start_time,
            segment.end_time,
            clip_path,
            include_audio=config.include_audio
        )

        if success:
            extracted_clips.append(clip_path)
            print(f"    -> Saved: {clip_filename}")
        else:
            print(f"    -> FAILED")

    stats.clips_extracted = len(extracted_clips)
    stats.extraction_time = time.time() - extraction_start

    print()
    print(f"Extraction Complete:")
    print(f"  Clips extracted: {stats.clips_extracted}/{len(segments)}")
    print(f"  Time: {stats.extraction_time:.1f}s")
    print()

    return extracted_clips


def save_manifest(
    clips: list[Path],
    segments: list[ClipSegment],
    config: ClipConfig,
    stats: ProcessingStats
):
    """Save manifest file with clip metadata."""
    manifest_path = config.get_manifest_path()

    with open(manifest_path, 'w') as f:
        f.write("Clip Extraction Manifest\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Source video: {config.video_path}\n")
        f.write(f"Reference folder: {config.reference_folder}\n")
        f.write(f"Clip padding: {config.clip_padding}s\n")
        f.write(f"Total clips: {len(clips)}\n\n")

        f.write("Processing Stats:\n")
        f.write(f"  Pass 1 time: {stats.pass1_time:.1f}s\n")
        f.write(f"  Pass 2 time: {stats.pass2_time:.1f}s\n")
        f.write(f"  Extraction time: {stats.extraction_time:.1f}s\n")
        f.write(f"  Total time: {stats.total_time:.1f}s\n\n")

        f.write("Clips:\n")
        f.write("-" * 50 + "\n")

        for i, (clip, segment) in enumerate(zip(clips, segments)):
            duration = segment.end_time - segment.start_time
            f.write(f"{i+1}. {clip.name}\n")
            f.write(f"   Time: {segment.start_formatted} - {segment.end_formatted}\n")
            f.write(f"   Duration: {duration:.1f}s\n")
            f.write(f"   Face matches: {segment.match_count}\n")
            f.write(f"   Peak similarity: {segment.peak_similarity:.3f}\n")
            f.write("\n")

    print(f"Manifest saved to: {manifest_path}")


def run_clip_extractor(config: ClipConfig) -> tuple[list[Path], ProcessingStats]:
    """Main entry point for clip extraction."""

    print("=" * 60)
    print("CLIP EXTRACTOR - Two-Pass Face Detection")
    print("=" * 60)
    print()

    overall_start = time.time()
    stats = ProcessingStats()

    # Check inputs
    if not Path(config.video_path).exists():
        print(f"Error: Video file not found: {config.video_path}")
        return [], stats

    if not Path(config.reference_folder).exists():
        print(f"Error: Reference folder not found: {config.reference_folder}")
        return [], stats

    # Get video info
    video_info = get_video_info(config.video_path)
    print(f"Video: {config.video_path}")
    print(f"  Duration: {format_timestamp(video_info['duration'])}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Codec: {video_info['codec']}")
    print(f"  Has audio: {video_info['has_audio']}")
    print(f"  Time budget: {config.max_processing_time}s")
    print(f"Output folder: {config.get_output_path()}")
    print()

    # Initialize face detection
    app = initialize_face_app()

    # Load reference faces
    reference_embeddings = load_reference_faces(config.reference_folder, app)
    if not reference_embeddings:
        print("Error: No reference faces loaded")
        return [], stats

    # Pass 1: Keyframe scan
    hot_segments, stats = pass1_keyframe_scan(
        config.video_path,
        reference_embeddings,
        app,
        config,
        overall_start
    )

    # Pass 2: Segment detection
    segments = pass2_segment_detection(
        config.video_path,
        hot_segments,
        reference_embeddings,
        app,
        config,
        stats,
        overall_start
    )

    # Extract clips
    clips = extract_all_clips(
        config.video_path,
        segments,
        config,
        stats
    )

    # Save manifest
    stats.total_time = time.time() - overall_start
    if clips:
        save_manifest(clips, segments[:len(clips)], config, stats)

    # Final summary
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total clips extracted: {len(clips)}")
    print(f"Total time: {stats.total_time:.1f}s")
    print(f"  Pass 1 (reconnaissance): {stats.pass1_time:.1f}s")
    print(f"  Pass 2 (detection): {stats.pass2_time:.1f}s")
    print(f"  Extraction: {stats.extraction_time:.1f}s")
    print()

    if clips:
        print("Extracted clips:")
        for clip in clips:
            print(f"  {clip.name}")

    return clips, stats


def main():
    """Example usage with default configuration."""

    config = ClipConfig(
        video_path="test.mp4",
        reference_folder="reference_images",
        output_folder="clips",
        output_relative_to_video=True,

        # Pass 1: Wide net
        pass1_scale=0.6,
        pass1_similarity_threshold=0.28,
        pass1_sample_between_keyframes=2,

        # Pass 2: Segment detection
        pass2_sample_interval=0.3,
        pass2_segment_padding=8.0,
        pass2_similarity_threshold=0.4,

        # Clip settings
        clip_padding=2.0,  # 2 seconds before/after
        min_clip_duration=1.0,
        include_audio=True,

        max_processing_time=1500,
    )

    clips, stats = run_clip_extractor(config)


if __name__ == "__main__":
    main()
