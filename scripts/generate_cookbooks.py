#!/usr/bin/env python3
"""Convert Jupyter notebooks in docs/cookbooks/ to GitBook-compatible .md pages.

Reads each .ipynb file, converts cells to markdown/code blocks, and writes the
result into docs/src/content/docs/cookbooks/. A Colab badge is injected at the
top of each generated page.

Usage:
    python scripts/generate_cookbooks.py          # Generate .md files
    python scripts/generate_cookbooks.py --check   # Verify .md files are up-to-date
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
COOKBOOKS_SRC = REPO_ROOT / "docs" / "cookbooks"
COOKBOOKS_DEST = REPO_ROOT / "docs" / "cookbooks"
GITHUB_BASE = "https://colab.research.google.com/github/mozilla-ai/any-llm/blob/main/docs/cookbooks"


def notebook_to_mdx(notebook_path: Path) -> str:
    """Convert a single .ipynb file to an .md string."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells:
        return ""

    # Extract title from first cell if it's a markdown heading
    title = notebook_path.stem.replace("_", " ").title()
    first_cell_source = "".join(cells[0].get("source", []))
    if first_cell_source.startswith("# "):
        title = first_cell_source.lstrip("# ").strip()

    colab_url = f"{GITHUB_BASE}/{notebook_path.name}"

    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: {title}")
    lines.append("---")
    lines.append("")
    lines.append(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})")
    lines.append("")

    for cell in cells:
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        cell_type = cell.get("cell_type", "code")

        if cell_type == "markdown":
            # Skip the title cell (already in frontmatter) and the Colab badge
            # cell (already injected above).
            stripped = source.strip()
            if stripped.startswith("# ") and cell is cells[0]:
                continue
            if "colab-badge.svg" in stripped:
                continue
            lines.append(source)
            lines.append("")
        elif cell_type == "code":
            lines.append("```python")
            lines.append(source)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def slug_for(notebook_path: Path) -> str:
    """Return the output .md filename for a notebook."""
    return notebook_path.stem.replace("_", "-") + ".md"


def generate_all() -> dict[Path, str]:
    """Generate .md content for every notebook. Returns {dest_path: content}."""
    results: dict[Path, str] = {}
    if not COOKBOOKS_SRC.exists():
        return results

    for nb_path in sorted(COOKBOOKS_SRC.glob("*.ipynb")):
        dest = COOKBOOKS_DEST / slug_for(nb_path)
        results[dest] = notebook_to_mdx(nb_path)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cookbook notebooks to .md")
    parser.add_argument("--check", action="store_true", help="Check if .md files are up-to-date")
    args = parser.parse_args()

    pages = generate_all()

    if not pages:
        print("No notebooks found in docs/cookbooks/", file=sys.stderr)
        return 1

    if args.check:
        for dest, content in pages.items():
            if not dest.exists():
                print(f"Missing: {dest}", file=sys.stderr)
                print("Run 'python scripts/generate_cookbooks.py' to generate cookbook pages.", file=sys.stderr)
                return 1
            if dest.read_text(encoding="utf-8") != content:
                print(f"Out of date: {dest}", file=sys.stderr)
                print("Run 'python scripts/generate_cookbooks.py' to regenerate cookbook pages.", file=sys.stderr)
                return 1
        print(f"✓ {len(pages)} cookbook page(s) up to date")
        return 0

    COOKBOOKS_DEST.mkdir(parents=True, exist_ok=True)
    for dest, content in pages.items():
        dest.write_text(content, encoding="utf-8")
        print(f"✓ {dest.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
