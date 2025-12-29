import type { Segment, VideoInfo, DetectionStatus, LoadResponse, ExportResponse } from '@/types';

const API_BASE = '';

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || `API error: ${res.status}`);
  }

  return res.json();
}

export async function loadVideo(
  videoFile: File | null,
  videoPath: string | null,
  referenceFiles: File[] | null,
  referenceFolder: string | null,
  clipPadding: number = 2.0
): Promise<LoadResponse> {
  const formData = new FormData();

  if (videoFile) {
    formData.append('video_file', videoFile);
  } else if (videoPath) {
    formData.append('video_path', videoPath);
  }

  if (referenceFiles && referenceFiles.length > 0) {
    referenceFiles.forEach((file) => {
      formData.append('reference_files', file);
    });
  } else if (referenceFolder) {
    formData.append('reference_folder', referenceFolder);
  }

  formData.append('clip_padding', clipPadding.toString());

  const res = await fetch(`${API_BASE}/api/load_upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || `Failed to load video: ${res.status}`);
  }

  return res.json();
}

export async function getDetectionStatus(): Promise<DetectionStatus> {
  return fetchApi<DetectionStatus>('/api/status');
}

export async function getSegments(): Promise<{ segments: Segment[] }> {
  return fetchApi<{ segments: Segment[] }>('/api/segments');
}

export async function getVideoInfo(): Promise<VideoInfo> {
  return fetchApi<VideoInfo>('/api/video/info');
}

export async function updateSegment(
  segmentId: string,
  updates: Partial<Pick<Segment, 'start_time' | 'end_time' | 'selected'>>
): Promise<Segment> {
  return fetchApi<Segment>(`/api/segments/${segmentId}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export async function addSegment(
  startTime: number,
  endTime: number
): Promise<Segment> {
  return fetchApi<Segment>('/api/segments', {
    method: 'POST',
    body: JSON.stringify({ start_time: startTime, end_time: endTime }),
  });
}

export async function deleteSegment(segmentId: string): Promise<void> {
  await fetchApi<{ status: string }>(`/api/segments/${segmentId}`, {
    method: 'DELETE',
  });
}

export async function exportClips(
  segmentIds: string[],
  outputFolder?: string
): Promise<ExportResponse> {
  return fetchApi<ExportResponse>('/api/export', {
    method: 'POST',
    body: JSON.stringify({
      segment_ids: segmentIds,
      output_folder: outputFolder,
    }),
  });
}

export function getVideoStreamUrl(): string {
  return `${API_BASE}/video/stream`;
}
