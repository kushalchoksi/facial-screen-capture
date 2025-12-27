import av
import cv2
import numpy as np
from pathlib import Path
import time
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: insightface not installed")
    print("Run: uv add insightface onnxruntime")
    exit(1)


@dataclass
class ProcessingConfig:
    """Configuration for the two-pass face finder."""
    # Input/Output
    video_path: str = "test.mp4"
    reference_folder: str = "reference_images"
    output_folder: str = "matched_frames"  # If relative, will be placed next to video
    output_relative_to_video: bool = True  # If True, output_folder is relative to video location
    
    # Pass 1 settings (reconnaissance)
    pass1_scale: float = 0.6  # Downscale for faster detection (was 0.5, too aggressive)
    pass1_similarity_threshold: float = 0.28  # Lower threshold - cast wide net (was 0.35)
    pass1_sample_between_keyframes: int = 2  # Also sample N frames between keyframes
    
    # Pass 2 settings (targeted capture)
    pass2_sample_interval: float = 0.3  # Sample every N seconds (was 1.0, too sparse)
    pass2_segment_padding: float = 8.0  # Seconds before/after keyframe (was 3.0)
    pass2_similarity_threshold: float = 0.4  # Final matching threshold
    pass2_min_gap_between_saves: float = 0.0  # Minimum gap between saved frames (0 = save all)
    
    # Time budget
    max_processing_time: int = 300  # 5 minutes in seconds
    
    # Output
    save_frames: bool = True
    timestamps_file: str = "timestamps.txt"  # Will also be placed with output_folder
    
    def get_output_path(self) -> Path:
        """Get the resolved output folder path."""
        if self.output_relative_to_video:
            video_dir = Path(self.video_path).parent
            return video_dir / self.output_folder
        return Path(self.output_folder)
    
    def get_timestamps_path(self) -> Path:
        """Get the resolved timestamps file path."""
        return self.get_output_path().parent / self.timestamps_file


@dataclass 
class MatchResult:
    """Single match result."""
    timestamp: float
    time_formatted: str
    similarity: float
    frame_image: Optional[np.ndarray] = None
    filename: Optional[str] = None


@dataclass
class ProcessingStats:
    """Track processing statistics."""
    pass1_keyframes_total: int = 0
    pass1_keyframes_with_faces: int = 0
    pass1_hot_segments: int = 0
    pass1_time: float = 0.0
    pass2_frames_processed: int = 0
    pass2_matches: int = 0
    pass2_time: float = 0.0
    total_time: float = 0.0


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    
    info = {
        'duration': float(container.duration / av.time_base) if container.duration else 0,
        'fps': float(stream.average_rate) if stream.average_rate else 24.0,
        'total_frames': stream.frames if stream.frames else 0,
        'width': stream.width,
        'height': stream.height,
        'codec': stream.codec_context.name,
    }
    
    container.close()
    return info


def pass1_keyframe_scan(
    video_path: str,
    reference_embeddings: list[np.ndarray],
    app: FaceAnalysis,
    config: ProcessingConfig,
    start_time: float
) -> tuple[list[dict], ProcessingStats]:
    """
    Pass 1: Fast scan of keyframes + sampled inter-keyframe frames.
    Returns timestamps of segments containing the target face.
    
    The inter-keyframe sampling helps catch faces that might be 
    obscured/blurred exactly at keyframe boundaries.
    """
    print("=" * 50)
    print("PASS 1: Keyframe Reconnaissance")
    print("=" * 50)
    
    stats = ProcessingStats()
    hot_segments = []
    pass1_start = time.time()
    
    # First, do keyframe-only pass
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"
    stream.thread_type = "AUTO"
    
    time_base = stream.time_base
    fps = float(stream.average_rate) if stream.average_rate else 24.0
    
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
    
    # Now build our sample points: keyframes + samples between them
    sample_points = []
    for i, kf_time in enumerate(keyframe_timestamps):
        sample_points.append(('keyframe', kf_time))
        
        # Add inter-keyframe samples
        if i < len(keyframe_timestamps) - 1 and config.pass1_sample_between_keyframes > 0:
            next_kf_time = keyframe_timestamps[i + 1]
            gap = next_kf_time - kf_time
            
            for j in range(1, config.pass1_sample_between_keyframes + 1):
                inter_time = kf_time + (gap * j / (config.pass1_sample_between_keyframes + 1))
                sample_points.append(('inter', inter_time))
    
    print(f"  Total sample points: {len(sample_points)} "
          f"({len(keyframe_timestamps)} keyframes + "
          f"{len(sample_points) - len(keyframe_timestamps)} inter-keyframe)")
    print()
    
    # Now process sample points
    print("  Phase 1b: Scanning for faces...")
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    # Don't skip frames this time - we need to seek to specific points
    
    processed_count = 0
    last_progress_time = time.time()
    last_seek_time = -999
    
    for sample_type, target_time in sample_points:
        # Time budget check
        elapsed = time.time() - start_time
        if elapsed > config.max_processing_time * 0.5:  # Use 50% of budget for Pass 1
            print(f"\n  Time budget reached for Pass 1 ({elapsed:.1f}s)")
            break
        
        # Progress update every 3 seconds
        if time.time() - last_progress_time > 3.0:
            print(f"  Scanning... {format_timestamp(target_time)} "
                  f"({processed_count}/{len(sample_points)} samples, {len(hot_segments)} hits)")
            last_progress_time = time.time()
        
        # Seek if we're not close to target (avoid seeking for every frame)
        if abs(target_time - last_seek_time) > 0.5:
            seek_pts = int(target_time / time_base)
            try:
                container.seek(seek_pts, stream=stream)
            except av.AVError:
                continue
        
        # Decode next frame
        try:
            frame = next(container.decode(stream))
        except (StopIteration, av.AVError):
            continue
        
        actual_time = float(frame.pts * time_base) if frame.pts else target_time
        last_seek_time = actual_time
        processed_count += 1
        
        if sample_type == 'keyframe':
            stats.pass1_keyframes_total += 1
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format='bgr24')
        
        # Downscale for faster processing
        if config.pass1_scale != 1.0:
            img = cv2.resize(img, None, 
                           fx=config.pass1_scale, 
                           fy=config.pass1_scale,
                           interpolation=cv2.INTER_LINEAR)
        
        # Detect faces
        faces = app.get(img)
        
        if faces:
            if sample_type == 'keyframe':
                stats.pass1_keyframes_with_faces += 1
            
            # Check against reference embeddings
            for face in faces:
                matched = False
                best_similarity = 0.0
                
                for ref_embedding in reference_embeddings:
                    similarity = compute_similarity(face.embedding, ref_embedding)
                    best_similarity = max(best_similarity, similarity)
                    
                    if similarity >= config.pass1_similarity_threshold:
                        # Avoid duplicate timestamps (within 0.3 sec)
                        if not any(abs(h['timestamp'] - actual_time) < 0.3 for h in hot_segments):
                            hot_segments.append({
                                'pts': frame.pts,
                                'timestamp': actual_time,
                                'similarity': similarity,
                                'source': sample_type
                            })
                            print(f"  [HIT] {format_timestamp(actual_time)} "
                                  f"(similarity: {similarity:.3f}, {sample_type})")
                        matched = True
                        break
                
                if matched:
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


def merge_adjacent_segments(segments: list[dict], max_gap: float = 10.0) -> list[tuple[float, float]]:
    """
    Merge adjacent hot segments to avoid redundant seeking.
    Returns list of (start_time, end_time) tuples.
    """
    if not segments:
        return []
    
    # Sort by timestamp
    sorted_segments = sorted(segments, key=lambda x: x['timestamp'])
    
    merged = []
    current_start = sorted_segments[0]['timestamp']
    current_end = sorted_segments[0]['timestamp']
    
    for seg in sorted_segments[1:]:
        if seg['timestamp'] - current_end <= max_gap:
            # Extend current segment
            current_end = seg['timestamp']
        else:
            # Start new segment
            merged.append((current_start, current_end))
            current_start = seg['timestamp']
            current_end = seg['timestamp']
    
    # Don't forget last segment
    merged.append((current_start, current_end))
    
    return merged


def pass2_targeted_capture(
    video_path: str,
    hot_segments: list[dict],
    reference_embeddings: list[np.ndarray],
    app: FaceAnalysis,
    config: ProcessingConfig,
    stats: ProcessingStats,
    start_time: float
) -> list[MatchResult]:
    """
    Pass 2: Dense sampling only in segments where target was detected.
    """
    print("=" * 50)
    print("PASS 2: Targeted Capture")
    print("=" * 50)
    
    if not hot_segments:
        print("  No hot segments to process")
        return []
    
    # Merge adjacent segments with generous gap allowance
    merged_segments = merge_adjacent_segments(hot_segments, max_gap=15.0)
    print(f"  Merged {len(hot_segments)} hits into {len(merged_segments)} segments")
    print(f"  Sample interval: {config.pass2_sample_interval}s")
    print(f"  Segment padding: {config.pass2_segment_padding}s")
    print(f"  Min gap between saves: {config.pass2_min_gap_between_saves}s")
    print()
        
    # Show segments we'll process
    print("  Segments to capture:")
    for i, (seg_start, seg_end) in enumerate(merged_segments):
        padded_start = max(0, seg_start - config.pass2_segment_padding)
        padded_end = seg_end + config.pass2_segment_padding
        print(f"    {i+1}. {format_timestamp(padded_start)} - {format_timestamp(padded_end)}")
    print()
    
    pass2_start = time.time()
    matches = []
    frames_processed = 0
    
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    time_base = stream.time_base
    fps = float(stream.average_rate) if stream.average_rate else 24.0
    
    # Calculate frame skip based on sample interval
    frame_skip = max(1, int(fps * config.pass2_sample_interval))
    
    for seg_idx, (seg_start, seg_end) in enumerate(merged_segments):
        # Time budget check
        elapsed = time.time() - start_time
        remaining = config.max_processing_time - elapsed
        if remaining < 10:
            print(f"\n  Time budget exhausted, stopping at segment {seg_idx + 1}/{len(merged_segments)}")
            break
        
        # Add padding
        actual_start = max(0, seg_start - config.pass2_segment_padding)
        actual_end = seg_end + config.pass2_segment_padding
        
        print(f"  Processing segment {seg_idx + 1}/{len(merged_segments)}: "
              f"{format_timestamp(actual_start)} - {format_timestamp(actual_end)}")
        
        # Seek to segment start
        seek_pts = int(actual_start / time_base)
        try:
            container.seek(seek_pts, stream=stream)
        except av.AVError:
            container.seek(0)
        
        frame_count_in_segment = 0
        last_match_time = -999
        segment_matches = 0
        
        for frame in container.decode(stream):
            current_time = float(frame.pts * time_base) if frame.pts else 0.0
            
            # Past segment end?
            if current_time > actual_end:
                break
            
            # Before segment start? (can happen after seek)
            if current_time < actual_start - 0.5:
                continue
            
            # Frame skip for sampling interval
            frame_count_in_segment += 1
            if frame_skip > 1 and frame_count_in_segment % frame_skip != 0:
                continue
            
            frames_processed += 1
            
            # Convert and detect
            img = frame.to_ndarray(format='bgr24')
            faces = app.get(img)
            
            for face in faces:
                for ref_embedding in reference_embeddings:
                    similarity = compute_similarity(face.embedding, ref_embedding)
                    
                    if similarity >= config.pass2_similarity_threshold:
                        # Check minimum gap between saves (0 = save all matches)
                        if config.pass2_min_gap_between_saves > 0:
                            if current_time - last_match_time < config.pass2_min_gap_between_saves:
                                continue
                        
                        last_match_time = current_time
                        segment_matches += 1
                        
                        match = MatchResult(
                            timestamp=current_time,
                            time_formatted=format_timestamp(current_time),
                            similarity=similarity,
                            frame_image=img.copy() if config.save_frames else None
                        )
                        matches.append(match)
                        
                        print(f"    [MATCH] {match.time_formatted} "
                              f"(similarity: {similarity:.3f})")
                        break
        
        print(f"    -> {segment_matches} matches in this segment")
    
    container.close()
    
    stats.pass2_frames_processed = frames_processed
    stats.pass2_matches = len(matches)
    stats.pass2_time = time.time() - pass2_start
    
    print()
    print(f"Pass 2 Complete:")
    print(f"  Frames processed: {stats.pass2_frames_processed}")
    print(f"  Matches found: {stats.pass2_matches}")
    print(f"  Time: {stats.pass2_time:.1f}s")
    print()
    
    return matches


def save_results(matches: list[MatchResult], config: ProcessingConfig, fps: float = 24.0) -> list[MatchResult]:
    """Save matched frames and timestamps."""
    if not matches:
        return matches
    
    output_path = config.get_output_path()
    timestamps_path = config.get_timestamps_path()
    
    if config.save_frames:
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(matches)} frames to {output_path}/")
        
        for match in matches:
            if match.frame_image is not None:
                # Include approximate frame number for easier comparison
                approx_frame = int(match.timestamp * fps)
                filename = f"frame_{approx_frame:06d}_{match.time_formatted.replace(':', '-')}_sim{match.similarity:.2f}.jpg"
                filepath = output_path / filename
                cv2.imwrite(str(filepath), match.frame_image)
                match.filename = filename
                # Clear frame from memory after saving
                match.frame_image = None
    
    # Save timestamps
    with open(timestamps_path, 'w') as f:
        f.write("Character Appearance Timestamps\n")
        f.write("=" * 40 + "\n\n")
        for match in matches:
            line = f"{match.time_formatted} (similarity: {match.similarity:.3f})"
            if match.filename:
                line += f" - {match.filename}"
            f.write(line + "\n")
    
    print(f"Timestamps saved to {timestamps_path}")
    
    return matches


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


def run_face_finder(config: ProcessingConfig) -> tuple[list[MatchResult], ProcessingStats]:
    """Main entry point for the two-pass face finder."""
    
    print("=" * 60)
    print("FAST FACE FINDER - Two-Pass Strategy")
    print("=" * 60)
    print()
    
    overall_start = time.time()
    
    # Check inputs
    if not Path(config.video_path).exists():
        print(f"Error: Video file not found: {config.video_path}")
        return [], ProcessingStats()
    
    if not Path(config.reference_folder).exists():
        print(f"Error: Reference folder not found: {config.reference_folder}")
        return [], ProcessingStats()
    
    # Get video info
    video_info = get_video_info(config.video_path)
    print(f"Video: {config.video_path}")
    print(f"  Duration: {format_timestamp(video_info['duration'])}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Codec: {video_info['codec']}")
    print(f"  Time budget: {config.max_processing_time}s")
    print(f"Output folder: {config.get_output_path()}")
    print()
    
    # Initialize face detection
    app = initialize_face_app()
    
    # Load reference faces
    reference_embeddings = load_reference_faces(config.reference_folder, app)
    if not reference_embeddings:
        print("Error: No reference faces loaded")
        return [], ProcessingStats()
    
    # Pass 1: Keyframe scan
    hot_segments, stats = pass1_keyframe_scan(
        config.video_path,
        reference_embeddings,
        app,
        config,
        overall_start
    )
    
    # Pass 2: Targeted capture
    matches = pass2_targeted_capture(
        config.video_path,
        hot_segments,
        reference_embeddings,
        app,
        config,
        stats,
        overall_start
    )
    
    # Save results
    matches = save_results(matches, config, fps=video_info['fps'])
    
    # Final stats
    stats.total_time = time.time() - overall_start
    
    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total matches: {len(matches)}")
    print(f"Total time: {stats.total_time:.1f}s")
    print(f"  Pass 1 (keyframes): {stats.pass1_time:.1f}s")
    print(f"  Pass 2 (targeted):  {stats.pass2_time:.1f}s")
    print()
    
    if matches:
        print("Timestamps:")
        for match in matches:
            print(f"  {match.time_formatted} (similarity: {match.similarity:.3f})")
    
    return matches, stats


# CLI entry point
def main():
    """Example usage with default configuration."""
    
    config = ProcessingConfig(
        # Video file - output will be saved next to this file
        video_path="test.mp4",
        
        # Reference images - can be anywhere on your system
        reference_folder="reference_images",
        
        # Output folder name (created next to video file)
        output_folder="matched_frames",
        output_relative_to_video=True,  # Set to False to use output_folder as absolute path
        
        # Pass 1: Cast a wide net
        pass1_scale=0.6,                    # 60% resolution (balance speed/accuracy)
        pass1_similarity_threshold=0.28,    # Low threshold - don't miss segments
        pass1_sample_between_keyframes=2,   # Sample 2 extra frames between keyframes
        
        # Pass 2: Dense capture in hot segments  
        pass2_sample_interval=0.3,          # Sample every 0.3 seconds
        pass2_segment_padding=8.0,          # 8 seconds padding around detections
        pass2_similarity_threshold=0.4,     # Final matching threshold
        pass2_min_gap_between_saves=0.0,    # Save ALL matches (no deduplication)
        
        max_processing_time=1500,
        save_frames=True,
    )
    
    matches, stats = run_face_finder(config)


if __name__ == "__main__":
    main()