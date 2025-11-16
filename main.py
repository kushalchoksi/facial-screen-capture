import face_recognition
import cv2
import os
from pathlib import Path


def load_reference_faces(sample_folder):
    """Load all sample images and create face encodings"""
    print(f"Loading reference images from {sample_folder}...")
    reference_encodings = []

    for image_file in Path(sample_folder).glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"  Processing {image_file.name}")
            image = face_recognition.load_image_file(str(image_file))
            encodings = face_recognition.face_encodings(image)

            if encodings:
                reference_encodings.append(encodings[0])
            else:
                print(f"  Warning: No face found in {image_file.name}")

    print(f"Loaded {len(reference_encodings)} reference face(s)\n")
    return reference_encodings


def find_character_in_video(video_path, reference_encodings, frame_skip=30, tolerance=0.6):
    """
    Process video and find timestamps where the character appears

    Args:
        video_path: Path to the video file
        reference_encodings: List of face encodings to match against
        frame_skip: Process every Nth frame (higher = faster but might miss appearances)
        tolerance: How strict the face matching is (lower = stricter, default 0.6)
    """
    print(f"Opening video: {video_path}")
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {fps} fps, {total_frames} frames")
    print(f"Processing every {frame_skip} frames")
    print(f"Matching tolerance: {tolerance}\n")

    matches = []
    frame_number = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Only process every Nth frame to speed things up
        if frame_number % frame_skip == 0:
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Check each face against reference faces
            for face_encoding in face_encodings:
                matches_found = face_recognition.compare_faces(
                    reference_encodings,
                    face_encoding,
                    tolerance=tolerance
                )

                if any(matches_found):
                    timestamp = frame_number / fps
                    matches.append({
                        'frame': frame_number,
                        'timestamp': timestamp,
                        'time_formatted': format_timestamp(timestamp)
                    })
                    print(f"** Match found at {format_timestamp(timestamp)} (frame {frame_number})")
                    break  # Found the character, move to next frame

        frame_number += 1

        # Progress indicator
        if frame_number % (frame_skip * 10) == 0:
            progress = (frame_number / total_frames) * 100
            print(f"Progress: {progress:.1f}%")

    video.release()

    print(f"\n{'='*50}")
    print(f"RESULTS: Found {len(matches)} appearances")
    print(f"{'='*50}")

    if matches:
        print("\nTimestamps where character appears:")
        for match in matches:
            print(f"  {match['time_formatted']} (frame {match['frame']})")

    return matches


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    sample_folder = "sample_images"
    video_file = "test.mp4"

    reference_encodings = load_reference_faces(sample_folder)

    if not reference_encodings:
        print("Error: No reference faces found. Please add sample images to the folder")
        return

    matches = find_character_in_video(
        video_file,
        reference_encodings,
        frame_skip=30,
        tolerance=0.6
    )

    if matches:
        with open('timestamps.txt', 'w') as f:
            f.write("Timestamps where character appears:\n\n")
            for match in matches:
                f.write(f"{match['time_formatted']}\n")
        print("\nTimestamps saved to timestamps.txt")


if __name__ == "__main__":
    main()
