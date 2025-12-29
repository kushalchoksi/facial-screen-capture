/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
      {
        source: '/video/:path*',
        destination: 'http://localhost:8000/video/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
