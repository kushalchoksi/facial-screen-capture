import av
from pathlib import Path
import traceback

# Hardcoded segment data from facial recognition run
# Format: (start_time, end_time, description)
TEST_SEGMENTS = [
    (13.0, 41.0, "segment_1"),   # 00:00:13 - 00:00:41 (27.6s)
    (61.0, 67.0, "segment_2"),   # 00:01:01 - 00:01:07 (6.3s)
    (68.0, 75.0, "segment_3"),   # 00:01:08 - 00:01:15 (7.2s)
    (77.0, 82.0, "segment_4"),   # 00:01:17 - 00:01:22 (5.2s)
]

VIDEO_PATH = "test.mp4"
OUTPUT_DIR = Path("clips")


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Testing extract_clip_pyav with {len(TEST_SEGMENTS)} segments")
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Just test with the first segment for now
    start_time, end_time, name = TEST_SEGMENTS[0]
    output_path = OUTPUT_DIR / f"test_{name}.mp4"

    print(f"Extracting: {start_time}s - {end_time}s -> {output_path}")

    success = extract_clip_pyav(
        VIDEO_PATH,
        start_time,
        end_time,
        output_path,
        include_audio=True
    )

    if success:
        print("SUCCESS!")
        # Verify the output
        if output_path.exists():
            container = av.open(str(output_path))
            duration = float(container.duration / av.time_base) if container.duration else 0
            print(f"  Output duration: {duration:.1f}s (expected: {end_time - start_time:.1f}s)")
            container.close()
    else:
        print("FAILED!")


if __name__ == "__main__":
    main()