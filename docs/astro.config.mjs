import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

const site = process.env.CI
  ? "https://mozilla-ai.github.io"
  : "http://localhost:4321";

export default defineConfig({
  site,
  base: "/any-llm",
  redirects: {
    "/api/any_llm/": "/api/any-llm/",
    "/api/list_models/": "/api/list-models/",
  },
  integrations: [
    starlight({
      title: "any-llm",
      logo: {
        src: "./src/assets/any-llm-logo-mark.png",
      },
      favicon: "/images/any-llm_favicon.png",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/mozilla-ai/any-llm",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      editLink: {
        baseUrl: "https://github.com/mozilla-ai/any-llm/edit/main/docs/",
      },
      sidebar: [
        { label: "Introduction", link: "/" },
        { label: "Quickstart", slug: "quickstart" },
        { label: "Providers", slug: "providers" },
        {
          label: "API Reference",
          items: [
            { label: "AnyLLM", slug: "api/any-llm" },
            { label: "Responses", slug: "api/responses" },
            { label: "Completion", slug: "api/completion" },
            { label: "Embedding", slug: "api/embedding" },
            { label: "Messages", slug: "api/messages" },
            { label: "Exceptions", slug: "api/exceptions" },
            { label: "List Models", slug: "api/list-models" },
            { label: "Batch", slug: "api/batch" },
            {
              label: "Types",
              items: [
                { label: "Completion", slug: "api/types/completion" },
                { label: "Responses", slug: "api/types/responses" },
                { label: "Messages", slug: "api/types/messages" },
                { label: "Model", slug: "api/types/model" },
                { label: "Provider", slug: "api/types/provider" },
                { label: "Batch", slug: "api/types/batch" },
              ],
            },
          ],
        },
        { label: "Managed Platform", slug: "platform/overview" },
        {
          label: "Gateway",
          items: [
            { label: "Overview", slug: "gateway/overview" },
            { label: "Quick Start", slug: "gateway/quickstart" },
            { label: "Authentication", slug: "gateway/authentication" },
            {
              label: "Budget Management",
              slug: "gateway/budget-management",
            },
            { label: "Configuration", slug: "gateway/configuration" },
            { label: "API Reference", slug: "gateway/api-reference" },
            { label: "Troubleshooting", slug: "gateway/troubleshooting" },
            {
              label: "Docker Deployment",
              slug: "gateway/docker-deployment",
            },
          ],
        },
      ],
      head: [
        {
          tag: "script",
          attrs: { type: "application/ld+json" },
          content: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareSourceCode",
            name: "any-llm",
            description:
              "A Python library providing a single interface to different LLM providers including OpenAI, Anthropic, Mistral, and more",
            programmingLanguage: "Python",
            codeRepository: "https://github.com/mozilla-ai/any-llm",
            license:
              "https://github.com/mozilla-ai/any-llm/blob/main/LICENSE",
            author: {
              "@type": "Organization",
              name: "Mozilla.ai",
              url: "https://mozilla.ai",
            },
          }),
        },
      ],
    }),
  ],
});
