import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_github_pages_redirects.py"


def test_generate_github_pages_redirects(tmp_path: Path) -> None:
    source_dir = tmp_path / "site"
    output_dir = tmp_path / "gh-pages-redirect"

    (source_dir / "api").mkdir(parents=True)
    (source_dir / "gateway").mkdir(parents=True)
    (source_dir / "images").mkdir(parents=True)

    (source_dir / "index.md").write_text("# Home\n", encoding="utf-8")
    (source_dir / "quickstart.md").write_text("# Quickstart\n", encoding="utf-8")
    (source_dir / "api" / "any-llm.md").write_text("# AnyLLM\n", encoding="utf-8")
    (source_dir / "gateway" / "overview.md").write_text("# Overview\n", encoding="utf-8")
    (source_dir / "SUMMARY.md").write_text("# Summary\n", encoding="utf-8")
    (source_dir / "openapi.json").write_text('{"openapi":"3.1.0"}\n', encoding="utf-8")
    (source_dir / "llms.txt").write_text("llms\n", encoding="utf-8")
    (source_dir / "images" / "logo.png").write_bytes(b"png")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    root_redirect = (output_dir / "index.html").read_text(encoding="utf-8")
    quickstart_redirect = (output_dir / "quickstart" / "index.html").read_text(encoding="utf-8")
    alias_redirect = (output_dir / "api" / "any_llm" / "index.html").read_text(encoding="utf-8")
    not_found_redirect = (output_dir / "404.html").read_text(encoding="utf-8")

    assert 'https://docs.mozilla.ai/any-llm/' in root_redirect
    assert 'https://docs.mozilla.ai/any-llm/quickstart/' in quickstart_redirect
    assert 'https://docs.mozilla.ai/any-llm/api/any-llm/' in alias_redirect
    assert (output_dir / "quickstart.html").exists()
    assert not (output_dir / "SUMMARY.html").exists()
    assert (output_dir / ".nojekyll").exists()
    assert (output_dir / "openapi.json").read_text(encoding="utf-8") == '{"openapi":"3.1.0"}\n'
    assert (output_dir / "llms.txt").read_text(encoding="utf-8") == "llms\n"
    assert (output_dir / "images" / "logo.png").read_bytes() == b"png"
    assert 'const pagesBasePath = "/any-llm";' in not_found_redirect
    assert 'const docsBaseUrl = new URL("https://docs.mozilla.ai/any-llm/");' in not_found_redirect
