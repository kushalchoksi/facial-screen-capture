import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Clip Editor',
  description: 'Video clipping and reference tool',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-dracula-bg text-dracula-fg antialiased">
        {children}
      </body>
    </html>
  );
}
