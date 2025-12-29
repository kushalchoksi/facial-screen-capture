'use client';

import { Header } from '@/components/Header';
import { Sidebar } from '@/components/Sidebar';
import { VideoPlayer } from '@/components/VideoPlayer';
import { Timeline } from '@/components/Timeline';
import { LoadModal } from '@/components/LoadModal';
import { useTimeline } from '@/hooks/useTimeline';

export default function Home() {
  // Initialize timeline hook for keyboard shortcuts
  useTimeline();

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <Header />

      {/* Main content area */}
      <div className="flex-1 flex min-h-0">
        {/* Sidebar */}
        <Sidebar />

        {/* Video player */}
        <main className="flex-1 flex flex-col min-w-0">
          <VideoPlayer />
        </main>
      </div>

      {/* Timeline */}
      <Timeline />

      {/* Load modal */}
      <LoadModal />
    </div>
  );
}
