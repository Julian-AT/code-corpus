import { ArticleHero } from "@/components/article-hero";
import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { TableOfContents } from "@/components/table-of-contents";
import Article from "@/content/report.mdx";
import {
  ARTICLE,
  HERO_IMAGE_URL,
  SITE_URL,
  TOC_SECTIONS,
} from "@/lib/constants";

const articleJsonLd = {
  "@context": "https://schema.org",
  "@type": "TechArticle",
  headline: ARTICLE.title,
  description: ARTICLE.description,
  datePublished: ARTICLE.datePublished,
  dateModified: ARTICLE.dateModified,
  author: ARTICLE.authors.map((name) => ({
    "@type": "Person",
    name,
    url: "https://julianschmidt.cv",
  })),
  image: new URL(HERO_IMAGE_URL, SITE_URL).toString(),
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
        <div className="mx-auto w-full max-w-[1400px] px-5 sm:px-8 lg:px-16">
          <ArticleHero />
          <div className="flex">
            <div className="w-0 shrink-0">
              <TableOfContents sections={[...TOC_SECTIONS]} />
            </div>
            <article className="mt-12 min-w-0 flex-1 pb-8">
              <Article />
            </article>
          </div>
        </div>
      </main>
      <SiteFooter />
    </>
  );
}
