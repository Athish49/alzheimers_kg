/** @type {import('next').NextConfig} */
const securityHeaders = [
  { key: 'X-Frame-Options',        value: 'SAMEORIGIN' },
  { key: 'X-Content-Type-Options', value: 'nosniff' },
  { key: 'Referrer-Policy',        value: 'strict-origin-when-cross-origin' },
  { key: 'Permissions-Policy',     value: 'camera=(), microphone=(), geolocation=()' },
  {
    key: 'Content-Security-Policy',
    value: [
      "default-src 'self'",
      "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  // unsafe-eval needed by Next.js dev
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' https://fonts.gstatic.com",
      "connect-src 'self'",
      "img-src 'self' data:",
      "frame-ancestors 'none'",
    ].join('; '),
  },
];

const nextConfig = {
  allowedDevOrigins: ['192.168.1.81'],

  async headers() {
    return [
      { source: '/(.*)', headers: securityHeaders },
    ];
  },

  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://localhost:8000';
    return [
      { source: '/answer', destination: `${backend}/answer` },
      { source: '/health', destination: `${backend}/health` },
    ];
  },
};

module.exports = nextConfig;
