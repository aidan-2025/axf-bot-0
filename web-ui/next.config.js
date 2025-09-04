/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  env: {
    APP1_API_URL: process.env.APP1_API_URL || 'http://localhost:8000',
    APP2_API_URL: process.env.APP2_API_URL || 'http://localhost:8001',
  },
  async rewrites() {
    return [
      {
        source: '/api/app1/:path*',
        destination: `${process.env.APP1_API_URL || 'http://localhost:8000'}/api/v1/:path*`,
      },
      {
        source: '/api/app2/:path*',
        destination: `${process.env.APP2_API_URL || 'http://localhost:8001'}/api/v1/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
