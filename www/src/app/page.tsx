import { ArticleHero } from "@/components/article-hero";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { TableOfContents } from "@/components/table-of-contents";
import Article from "@/content/report.mdx";
import { ARTICLE, SITE_URL, TOC_SECTIONS } from "@/lib/constants";

const articleJsonLd = {
  "@context": "https://schema.org",
  "@type": "TechArticle",
  headline: ARTICLE.title,
  description: ARTICLE.description,
  datePublished: ARTICLE.datePublished,
  dateModified: ARTICLE.dateModified,
  author: {
    "@type": "Person",
    name: "Julian Schmidt",
    url: "https://julianschmidt.cv",
    sameAs: ["https://github.com/Julian-AT"],
  },
  about: ["Gemma 4", "MLX LoRA", "Ollama", "personal code corpus"],
  mainEntityOfPage: SITE_URL,
};

export default function Home() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
      <SiteHeader />
      <main id="main-content">
        <ArticleHero />
        <div className="report-shell mx-auto w-full max-w-[1440px] px-5 sm:px-8 lg:px-12">
          <div className="flex items-start gap-10 xl:gap-16">
            <TableOfContents sections={[...TOC_SECTIONS]} />
            <article className="min-w-0 flex-1 pb-24 pt-14 lg:pt-20">
              <Article />
            </article>
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
