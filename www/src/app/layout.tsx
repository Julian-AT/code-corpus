import type { Metadata, Viewport } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { ARTICLE, SITE_URL } from "@/lib/constants";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: `${ARTICLE.title} | Julian Schmidt`,
    template: "%s | Julian Schmidt",
  },
  description: ARTICLE.description,
  applicationName: "Julian local model report",
  authors: [{ name: "Julian Schmidt", url: "https://julianschmidt.cv" }],
  creator: "Julian Schmidt",
  keywords: [
    "Gemma 4",
    "MLX",
    "LoRA",
    "Ollama",
    "Codex",
    "code corpus",
    "local AI",
  ],
  alternates: { canonical: "/" },
  openGraph: {
    title: ARTICLE.title,
    description: ARTICLE.description,
    url: "/",
    type: "article",
    publishedTime: ARTICLE.datePublished,
    modifiedTime: ARTICLE.dateModified,
    authors: ["Julian Schmidt"],
  },
  twitter: {
    card: "summary",
    title: ARTICLE.title,
    description: ARTICLE.description,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#f3f7f8",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} h-full`}
    >
      <body className={`${GeistSans.className} min-h-full`}>{children}</body>
    </html>
  );
}
