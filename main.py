import cv2
import numpy as np
from pathlib import Path
import os

try:
    from insightface.app import FaceAnalysis
except ImportError:
    exit(1)


def load_reference_faces_gpu(sample_folder, app):
    print(f"Loading reference images from {sample_folder}...")
    reference_embeddings = []

    for image_file in Path(sample_folder).glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"  Processing {image_file.name}")
            img = cv2.imread(str(image_file))

            if img is None:
                print(f"  Warning: Could not read {image_file.name}")
                continue

            faces = app.get(img)

            if faces:
                # Use the largest face (by bounding box area)
                largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                reference_embeddings.append(largest_face.embedding)
                print(f"   Face detected")
            else:
                print(f"  Warning: No face found in {image_file.name}")

    print(f"Loaded {len(reference_embeddings)} reference face(s)\n")
    return reference_embeddings


def compute_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def find_character_in_video_gpu(video_path, reference_embeddings, app,
                                frame_skip=5, similarity_threshold=0.4,
                                save_frames=True, output_folder="matched_frames"):
    print(f"Opening video: {video_path}")
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {fps} fps, {total_frames} frames")
    print(f"Processing every {frame_skip} frames")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Using GPU acceleration CUDA enabled\n")

    if save_frames:
        Path(output_folder).mkdir(exist_ok=True)
        print(f"Frames will be saved to: {output_folder}/\n")

    matches = []
    frame_number = 0
    processed_count = 0

    print("Processing video...")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            faces = app.get(frame)

            for face in faces:
                for ref_embedding in reference_embeddings:
                    similarity = compute_similarity(face.embedding, ref_embedding)

                    if similarity >= similarity_threshold:
                        timestamp = frame_number / fps

                        match_info = {
                            'frame': frame_number,
                            'timestamp': timestamp,
                            'time_formatted': format_timestamp(timestamp),
                            'similarity': similarity,
                            'frame_image': frame.copy() if save_frames else None
                        }

                        matches.append(match_info)
                        print(f"[MATCH] Found at {format_timestamp(timestamp)} "
                              f"(frame {frame_number}, similarity: {similarity:.3f})")
                        break

            processed_count += 1

            if processed_count % 50 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({processed_count} frames processed)")

        frame_number += 1

    video.release()

    results = []
    if save_frames and matches:
        print(f"\nSaving {len(matches)} matched frames...")
        for match in matches:
            filename = f"frame_{match['frame']:06d}_{match['time_formatted'].replace(':', '-')}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, match['frame_image'])

            results.append({
                'frame': match['frame'],
                'timestamp': match['timestamp'],
                'time_formatted': match['time_formatted'],
                'similarity': match['similarity'],
                'filename': filename
            })
        print(f"Frames saved to {output_folder}/")
    else:
        results = [{k: v for k, v in m.items() if k != 'frame_image'} for m in matches]

    print(f"\n{'='*50}")
    print(f"RESULTS: Found {len(results)} appearances")
    print(f"{'='*50}")

    if results:
        print("\nTimestamps where character appears:")
        for match in results:
            print(f"  {match['time_formatted']} (frame {match['frame']}, "
                  f"similarity: {match['similarity']:.3f})")

    return results


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    sample_folder = "sample_images"
    video_file = "test.mp4"

    frame_skip = 10
    similarity_threshold = 0.4

    save_frames = True
    output_folder = "matched_frames"

    print("="*50)
    print("GPU-ACCELERATED FACE FINDER")
    print("="*50 + "\n")

    print("Initializing GPU face detection...")
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for first GPU
        print("GPU initialized successfully!\n")
    except Exception as e:
        print(f"Warning: GPU initialization failed: {e}")
        print("Falling back to CPU...\n")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))

    reference_embeddings = load_reference_faces_gpu(sample_folder, app)

    if not reference_embeddings:
        print("Error: No reference faces found.")
        return

    matches = find_character_in_video_gpu(
        video_file,
        reference_embeddings,
        app,
        frame_skip=frame_skip,
        similarity_threshold=similarity_threshold,
        save_frames=save_frames,
        output_folder=output_folder
    )

    if matches:
        with open('timestamps.txt', 'w') as f:
            f.write("Timestamps where character appears:\n\n")
            for match in matches:
                line = f"{match['time_formatted']} (similarity: {match['similarity']:.3f})"
                if 'filename' in match:
                    line += f" - {match['filename']}"
                f.write(line + "\n")
        print("\nTimestamps saved to timestamps.txt")


if __name__ == "__main__":
    main()
