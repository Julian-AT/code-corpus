import createMDX from "@next/mdx";

const nextConfig = {
  pageExtensions: ["ts", "tsx", "js", "jsx", "md", "mdx"],
  transpilePackages: ["geist"],
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "www.julianschmidt.cv",
        pathname: "/_next/image",
      },
      {
        protocol: "https",
        hostname: "media.licdn.com",
        pathname: "/dms/image/**",
      },
    ],
  },
  turbopack: {
    root: import.meta.dirname,
  },
};

const withMDX = createMDX({
  options: {
    rehypePlugins: ["rehype-slug"],
  },
});

export default withMDX(nextConfig);
