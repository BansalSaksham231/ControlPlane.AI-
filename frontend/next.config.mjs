/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emit a self-contained server bundle for a slim production container.
  output: "standalone",
};

export default nextConfig;
