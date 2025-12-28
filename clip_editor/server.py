import shutil
import av
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mimetypes
import os
import traceback
import threading

from .detection import get_detector, DetectedSegment, DetectionProgress


app = FastAPI(title="Clip Editor API")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- State ---

class AppState:
    """Application state."""
    def __init__(self):
        self.video_path: Optional[str] = None
        self.reference_folder: Optional[str] = None
        self.segments: list[DetectedSegment] = []
        self.detection_progress: Optional[DetectionProgress] = None
        self.detection_running: bool = False
        self.video_info: Optional[dict] = None
        self.lock = threading.Lock()


state = AppState()


# --- Pydantic Models ---

class LoadVideoRequest(BaseModel):
    video_path: str
    reference_folder: str
    clip_padding: float = 2.0


class UpdateSegmentRequest(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    selected: Optional[bool] = None


class AddSegmentRequest(BaseModel):
    start_time: float
    end_time: float


class ExportRequest(BaseModel):
    segment_ids: list[str]
    output_folder: Optional[str] = None


# --- Helper Functions ---

def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    container = av.open(video_path)
    stream = container.streams.video[0]

    info = {
        'duration': float(container.duration / av.time_base) if container.duration else 0,
        'fps': float(stream.average_rate) if stream.average_rate else 24.0,
        'width': stream.width,
        'height': stream.height,
        'codec': stream.codec_context.name,
        'has_audio': len(container.streams.audio) > 0,
    }

    container.close()
    return info


def run_detection(video_path: str, reference_folder: str, clip_padding: float):
    """Run detection in background thread."""
    try:
        detector = get_detector()

        # Load reference images
        state.detection_progress = DetectionProgress("loading", 0.0, "Loading reference images...")
        count = detector.load_reference_images(reference_folder)

        if count == 0:
            state.detection_progress = DetectionProgress("error", 0.0, "No reference faces found")
            state.detection_running = False
            return

        # Run detection
        def progress_callback(progress: DetectionProgress):
            state.detection_progress = progress

        segments = detector.detect_segments(
            video_path,
            progress_callback=progress_callback,
            clip_padding=clip_padding,
        )

        with state.lock:
            state.segments = segments
            state.detection_running = False

    except Exception as e:
        state.detection_progress = DetectionProgress("error", 0.0, str(e))
        state.detection_running = False


def extract_clip(video_path: str, start_time: float, end_time: float, output_path: Path) -> bool:
    """Extract a clip using PyAV with stream copy."""
    try:
        with av.open(video_path) as in_container:
            out_container = av.open(str(output_path), mode='w')

            in_streams = []
            out_streams = []
            for in_stream in in_container.streams:
                if in_stream.type in ('video', 'audio'):
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
        print(f"Error extracting clip: {e}")
        return False

def start_video_load(video_path: Path, reference_folder: Path, clip_padding: float):
    """Shared logic to start video loading and detection."""
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {video_path}")

    if not reference_folder.exists():
        raise HTTPException(404, f"Reference folder not found: {reference_folder}")

    # Get video info
    try:
        info = get_video_info(str(video_path))
    except Exception as e:
        raise HTTPException(400, f"Failed to read video: {e}")

    # Update state
    with state.lock:
        state.video_path = str(video_path.resolve())
        state.reference_folder = str(reference_folder.resolve())
        state.video_info = info
        state.segments = []
        state.detection_running = True
        state.detection_progress = DetectionProgress("loading", 0.0, "Starting...")

    # Start detection in background
    thread = threading.Thread(
        target=run_detection,
        args=(state.video_path, state.reference_folder, clip_padding)
    )
    thread.start()

    return {
        "status": "ok",
        "video_info": info,
    }


# --- API Endpoints ---

@app.post("/api/load")
async def load_video(request: LoadVideoRequest, background_tasks: BackgroundTasks):
    """Load a video from path and start face detection."""
    return start_video_load(Path(request.video_path), Path(request.reference_folder), request.clip_padding)


@app.post("/api/load_upload")
async def load_video_upload(
    video_file: Optional[UploadFile] = File(default=None),
    video_path: Optional[str] = Form(default=None),
    reference_files: Optional[List[UploadFile]] = File(default=None),
    reference_folder: Optional[str] = Form(default=None),
    clip_padding: float = Form(default=2.0)
):
    """Upload video and/or reference images and start detection."""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # 1. Handle Video
    final_video_path = None
    if video_file:
        # Save uploaded video
        final_video_path = uploads_dir / video_file.filename
        try:
            with final_video_path.open("wb") as buffer:
                shutil.copyfileobj(video_file.file, buffer)
        except Exception as e:
            raise HTTPException(500, f"Failed to save uploaded video: {e}")
    elif video_path:
        final_video_path = Path(video_path)
    else:
        raise HTTPException(400, "Must provide either video_file or video_path")

    # 2. Handle References
    final_ref_folder = None
    if reference_files:
        # Create unique folder for this batch of references
        ref_id = uuid.uuid4().hex
        final_ref_folder = uploads_dir / "refs" / ref_id
        final_ref_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            for ref_file in reference_files:
                # Skip empty filenames (can happen with directory upload sometimes)
                if not ref_file.filename:
                    continue
                
                # We only care about the filename, not the relative path structure for now
                # (flattening the directory structure)
                safe_name = Path(ref_file.filename).name
                dest_path = final_ref_folder / safe_name
                
                with dest_path.open("wb") as buffer:
                    shutil.copyfileobj(ref_file.file, buffer)
        except Exception as e:
            raise HTTPException(500, f"Failed to save reference images: {e}")
            
    elif reference_folder:
        final_ref_folder = Path(reference_folder)
    else:
        raise HTTPException(400, "No reference files received (check your folder selection) and no server path provided.")

    return start_video_load(final_video_path, final_ref_folder, clip_padding)


@app.get("/api/status")
async def get_status():
    """Get detection progress."""
    return {
        "running": state.detection_running,
        "progress": state.detection_progress.progress if state.detection_progress else 0,
        "phase": state.detection_progress.phase if state.detection_progress else "idle",
        "message": state.detection_progress.message if state.detection_progress else "",
    }


@app.get("/api/segments")
async def get_segments():
    """Get all detected segments."""
    with state.lock:
        return {
            "segments": [s.to_dict() for s in state.segments]
        }


@app.put("/api/segments/{segment_id}")
async def update_segment(segment_id: str, request: UpdateSegmentRequest):
    """Update a segment's boundaries or selection."""
    with state.lock:
        for seg in state.segments:
            if seg.id == segment_id:
                if request.start_time is not None:
                    seg.start_time = request.start_time
                if request.end_time is not None:
                    seg.end_time = request.end_time
                if request.selected is not None:
                    seg.selected = request.selected
                return seg.to_dict()

    raise HTTPException(404, f"Segment not found: {segment_id}")


@app.delete("/api/segments/{segment_id}")
async def delete_segment(segment_id: str):
    """Delete a segment."""
    with state.lock:
        for i, seg in enumerate(state.segments):
            if seg.id == segment_id:
                state.segments.pop(i)
                return {"status": "ok"}

    raise HTTPException(404, f"Segment not found: {segment_id}")


@app.post("/api/segments")
async def add_segment(request: AddSegmentRequest):
    """Add a new segment manually."""
    with state.lock:
        new_id = f"seg_{len(state.segments)+1:03d}_{uuid.uuid4().hex[:4]}"
        segment = DetectedSegment(
            id=new_id,
            start_time=request.start_time,
            end_time=request.end_time,
            match_count=0,
            peak_similarity=0.0,
            selected=True,
        )
        state.segments.append(segment)
        # Re-sort by start time
        state.segments.sort(key=lambda s: s.start_time)
        return segment.to_dict()


@app.post("/api/export")
async def export_clips(request: ExportRequest):
    """Export selected segments as video clips."""
    if not state.video_path:
        raise HTTPException(400, "No video loaded")

    # Determine output folder
    if request.output_folder:
        output_folder = Path(request.output_folder)
    else:
        output_folder = Path(state.video_path).parent / "clips"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Find segments to export
    with state.lock:
        segments_to_export = [s for s in state.segments if s.id in request.segment_ids]

    if not segments_to_export:
        raise HTTPException(400, "No segments selected for export")

    video_ext = Path(state.video_path).suffix
    exported = []

    for seg in segments_to_export:
        start_str = f"{int(seg.start_time//3600):02d}-{int((seg.start_time%3600)//60):02d}-{int(seg.start_time%60):02d}"
        end_str = f"{int(seg.end_time//3600):02d}-{int((seg.end_time%3600)//60):02d}-{int(seg.end_time%60):02d}"
        filename = f"clip_{seg.id}_{start_str}_to_{end_str}{video_ext}"
        output_path = output_folder / filename

        success = extract_clip(state.video_path, seg.start_time, seg.end_time, output_path)
        if success:
            exported.append(str(output_path))

    return {
        "status": "ok",
        "exported": exported,
        "output_folder": str(output_folder),
    }


@app.get("/api/video/info")
async def get_video_info_endpoint():
    """Get current video info."""
    if not state.video_info:
        raise HTTPException(400, "No video loaded")
    return state.video_info


@app.get("/video/stream")
async def stream_video(request: Request, range: str = Header(default=None)):
    """
    Stream the loaded video file with HTTP Range support.
    This enables efficient seeking in large video files without
    loading the entire file into memory.
    """
    if not state.video_path:
        raise HTTPException(400, "No video loaded")

    video_path = Path(state.video_path)
    if not video_path.exists():
        raise HTTPException(404, "Video file not found")

    file_size = video_path.stat().st_size

    # Determine content type
    content_type, _ = mimetypes.guess_type(str(video_path))
    if not content_type:
        content_type = "video/mp4"

    # Handle range request for partial content
    if range:
        # Parse range header: "bytes=start-end" or "bytes=start-"
        range_str = range.replace("bytes=", "")
        parts = range_str.split("-")

        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1

        # Clamp to file bounds
        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_file_range():
            """Generator that yields chunks from the specified range."""
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file_range(),
            status_code=206,  # Partial Content
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
        )

    # No range requested - stream entire file
    def iter_file():
        """Generator that yields the entire file in chunks."""
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(video_path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        }
    )


@app.get("/frame/{timestamp}")
async def get_frame(timestamp: float):
    """Get a single frame as JPEG at the given timestamp."""
    if not state.video_path:
        raise HTTPException(400, "No video loaded")

    try:
        container = av.open(state.video_path)
        stream = container.streams.video[0]
        time_base = stream.time_base

        seek_pts = int(timestamp / time_base)
        container.seek(seek_pts, stream=stream)

        frame = next(container.decode(stream))
        img = frame.to_ndarray(format='rgb24')

        container.close()

        # Encode as JPEG
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg"
        )

    except Exception as e:
        raise HTTPException(500, f"Failed to extract frame: {e}")


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
