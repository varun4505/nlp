import type { NextConfig } from "next";

// The Python backend (api/server.py) runs on port 8000; proxy /api/* to it so the
// browser only ever talks to the Next.js origin.
const API = process.env.DRIFTLENS_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
