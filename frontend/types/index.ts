export interface Segment {
  id: string;
  start_time: number;
  end_time: number;
  match_count: number;
  peak_similarity: number;
  selected: boolean;
}

export interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  codec: string;
  has_audio: boolean;
}

export interface DetectionStatus {
  running: boolean;
  progress: number;
  phase: 'loading' | 'pass1' | 'pass2' | 'complete' | 'error' | 'idle';
  message: string;
}

export interface LoadResponse {
  status: string;
  video_info: VideoInfo;
}

export interface ExportResponse {
  status: string;
  exported: string[];
  output_folder: string;
}
