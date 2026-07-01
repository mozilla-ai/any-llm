"""Build a redirect-only GitHub Pages site for legacy any-llm docs URLs.

Usage:
    python scripts/generate_github_pages_redirects.py
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

DEFAULT_SOURCE_DIR = Path("site")
DEFAULT_OUTPUT_DIR = Path("gh-pages-redirect")
DEFAULT_BASE_URL = "https://docs.mozilla.ai/any-llm/"
DEFAULT_PAGES_BASE_PATH = "/any-llm"
PASSTHROUGH_FILES = ("openapi.json", "llms.txt", "llms-full.txt")
PASSTHROUGH_DIRS = ("images",)


def normalize_base_url(base_url: str) -> str:
    """Return the destination docs URL with a single trailing slash."""
    return base_url.rstrip("/") + "/"


def normalize_pages_base_path(pages_base_path: str) -> str:
    """Return the GitHub Pages project prefix with a leading slash."""
    stripped = pages_base_path.strip("/")
    return f"/{stripped}" if stripped else "/"


def route_for_markdown_file(relative_path: Path) -> str:
    """Convert a markdown file path into a route path."""
    route_parts = list(relative_path.with_suffix("").parts)
    if route_parts and route_parts[-1] == "index":
        route_parts = route_parts[:-1]
    return "/".join(route_parts)


def legacy_route_aliases(route: str) -> set[str]:
    """Return route aliases for legacy slug variants."""
    if not route:
        return set()

    route_parts = route.split("/")
    last_segment = route_parts[-1]
    aliases = set()

    for alias_segment in {
        last_segment.replace("-", "_"),
        last_segment.replace("_", "-"),
    }:
        if alias_segment and alias_segment != last_segment:
            aliases.add("/".join([*route_parts[:-1], alias_segment]))

    return aliases


def collect_redirect_targets(source_dir: Path) -> dict[str, str]:
    """Collect route-to-target mappings for markdown-backed pages."""
    routes: dict[str, str] = {}

    for markdown_file in source_dir.rglob("*.md"):
        relative_path = markdown_file.relative_to(source_dir)
        if relative_path.name == "SUMMARY.md":
            continue

        canonical_route = route_for_markdown_file(relative_path)
        routes[canonical_route] = canonical_route

        for alias_route in legacy_route_aliases(canonical_route):
            routes[alias_route] = canonical_route

    return routes


def build_target_url(base_url: str, route: str) -> str:
    """Build the redirect destination for a route."""
    if not route:
        return base_url

    normalized_route = quote(route.strip("/"), safe="/")
    return f"{base_url}{normalized_route}/"


def redirect_page_html(target_url: str) -> str:
    """Build a simple HTML redirect page."""
    escaped_target_url = html.escape(target_url, quote=True)
    serialized_target_url = json.dumps(target_url)

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <meta name="robots" content="noindex">
    <link rel="canonical" href="{escaped_target_url}">
    <meta http-equiv="refresh" content="0; url={escaped_target_url}">
    <script>
      const target = new URL({serialized_target_url});
      target.search = window.location.search;
      target.hash = window.location.hash;
      window.location.replace(target.toString());
    </script>
  </head>
  <body>
    <p>This documentation moved to <a href="{escaped_target_url}">{escaped_target_url}</a>.</p>
  </body>
</html>
"""


def not_found_page_html(base_url: str, pages_base_path: str) -> str:
    """Build a smart 404 page that preserves deep links."""
    escaped_base_url = html.escape(base_url, quote=True)
    serialized_base_url = json.dumps(base_url)
    serialized_pages_base_path = json.dumps(pages_base_path)

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Redirecting...</title>
    <meta name="robots" content="noindex">
    <meta http-equiv="refresh" content="0; url={escaped_base_url}">
    <script>
      const docsBaseUrl = new URL({serialized_base_url});
      const pagesBasePath = {serialized_pages_base_path};

      function normalizeRelativePath(pathname) {{
        let relativePath = pathname;

        if (relativePath === pagesBasePath) {{
          relativePath = "/";
        }} else if (relativePath.startsWith(`${{pagesBasePath}}/`)) {{
          relativePath = relativePath.slice(pagesBasePath.length);
        }}

        if (relativePath.endsWith("/index.html")) {{
          relativePath = relativePath.slice(0, -"/index.html".length) || "/";
        }} else if (relativePath.endsWith(".html")) {{
          relativePath = `${{relativePath.slice(0, -".html".length)}}/`;
        }}

        if (!relativePath.startsWith("/")) {{
          relativePath = `/${{relativePath}}`;
        }}

        if (relativePath !== "/" && !relativePath.endsWith("/") && !/\\.[^/]+$/.test(relativePath)) {{
          relativePath = `${{relativePath}}/`;
        }}

        return relativePath;
      }}

      const relativePath = normalizeRelativePath(window.location.pathname);
      const target = new URL(relativePath.replace(/^\\//, ""), docsBaseUrl);
      target.search = window.location.search;
      target.hash = window.location.hash;
      window.location.replace(target.toString());
    </script>
  </head>
  <body>
    <p>This documentation moved to <a href="{escaped_base_url}">{escaped_base_url}</a>.</p>
  </body>
</html>
"""


def write_file(path: Path, content: str) -> None:
    """Write a UTF-8 text file, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_passthrough_assets(source_dir: Path, output_dir: Path) -> None:
    """Copy machine-readable artifacts that should stay directly accessible."""
    for file_name in PASSTHROUGH_FILES:
        source_path = source_dir / file_name
        if not source_path.exists():
            continue

        destination_path = output_dir / file_name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    for directory_name in PASSTHROUGH_DIRS:
        source_path = source_dir / directory_name
        if not source_path.exists():
            continue

        shutil.copytree(source_path, output_dir / directory_name)


def build_redirect_site(
    source_dir: Path,
    output_dir: Path,
    base_url: str,
    pages_base_path: str,
) -> int:
    """Build the redirect site and return the number of redirect pages."""
    normalized_base_url = normalize_base_url(base_url)
    normalized_pages_base_path = normalize_pages_base_path(pages_base_path)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    redirect_targets = collect_redirect_targets(source_dir)
    redirect_count = 0

    for route, target_route in sorted(redirect_targets.items()):
        redirect_html = redirect_page_html(build_target_url(normalized_base_url, target_route))

        if route:
            directory_redirect_path = output_dir / route / "index.html"
            file_redirect_path = output_dir / Path(route).with_suffix(".html")
            write_file(directory_redirect_path, redirect_html)
            write_file(file_redirect_path, redirect_html)
            redirect_count += 2
        else:
            write_file(output_dir / "index.html", redirect_html)
            redirect_count += 1

    write_file(output_dir / "404.html", not_found_page_html(normalized_base_url, normalized_pages_base_path))
    write_file(output_dir / ".nojekyll", "")
    copy_passthrough_assets(source_dir, output_dir)

    return redirect_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Markdown site input directory. Defaults to {DEFAULT_SOURCE_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Redirect site output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Destination docs base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--pages-base-path",
        default=DEFAULT_PAGES_BASE_PATH,
        help=f"GitHub Pages project path prefix. Defaults to {DEFAULT_PAGES_BASE_PATH}.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the redirect site for legacy GitHub Pages routes."""
    args = parse_args()

    if not args.source_dir.exists():
        print(f"Input directory does not exist: {args.source_dir}", file=sys.stderr)
        return 1

    redirect_count = build_redirect_site(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        base_url=args.base_url,
        pages_base_path=args.pages_base_path,
    )
    print(f"Done - {redirect_count} redirect pages written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
