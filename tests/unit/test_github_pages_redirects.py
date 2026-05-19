import importlib.util
import runpy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_github_pages_redirects.py"


def _load_redirect_generator_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_github_pages_redirects", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sample_site(source_dir: Path, *, include_passthroughs: bool = True, include_images: bool = True) -> None:
    (source_dir / "api").mkdir(parents=True)
    (source_dir / "gateway").mkdir(parents=True)
    if include_images:
        (source_dir / "images").mkdir(parents=True)

    (source_dir / "index.md").write_text("# Home\n", encoding="utf-8")
    (source_dir / "quickstart.md").write_text("# Quickstart\n", encoding="utf-8")
    (source_dir / "api" / "any-llm.md").write_text("# AnyLLM\n", encoding="utf-8")
    (source_dir / "gateway" / "overview.md").write_text("# Overview\n", encoding="utf-8")
    (source_dir / "SUMMARY.md").write_text("# Summary\n", encoding="utf-8")

    if include_passthroughs:
        (source_dir / "openapi.json").write_text('{"openapi":"3.1.0"}\n', encoding="utf-8")
        (source_dir / "llms.txt").write_text("llms\n", encoding="utf-8")

    if include_images:
        (source_dir / "images" / "logo.png").write_bytes(b"png")


def test_generate_github_pages_redirects(tmp_path: Path) -> None:
    source_dir = tmp_path / "site"
    output_dir = tmp_path / "gh-pages-redirect"
    redirect_generator = _load_redirect_generator_module()

    _write_sample_site(source_dir)

    redirect_count = redirect_generator.build_redirect_site(
        source_dir=source_dir,
        output_dir=output_dir,
        base_url=redirect_generator.DEFAULT_BASE_URL,
        pages_base_path=redirect_generator.DEFAULT_PAGES_BASE_PATH,
    )

    root_redirect = (output_dir / "index.html").read_text(encoding="utf-8")
    quickstart_redirect = (output_dir / "quickstart" / "index.html").read_text(encoding="utf-8")
    alias_redirect = (output_dir / "api" / "any_llm" / "index.html").read_text(encoding="utf-8")
    not_found_redirect = (output_dir / "404.html").read_text(encoding="utf-8")

    assert redirect_count > 0
    assert "https://docs.mozilla.ai/any-llm/" in root_redirect
    assert "https://docs.mozilla.ai/any-llm/quickstart/" in quickstart_redirect
    assert "https://docs.mozilla.ai/any-llm/api/any-llm/" in alias_redirect
    assert (output_dir / "quickstart.html").exists()
    assert not (output_dir / "SUMMARY.html").exists()
    assert (output_dir / ".nojekyll").exists()
    assert (output_dir / "openapi.json").read_text(encoding="utf-8") == '{"openapi":"3.1.0"}\n'
    assert (output_dir / "llms.txt").read_text(encoding="utf-8") == "llms\n"
    assert (output_dir / "images" / "logo.png").read_bytes() == b"png"
    assert 'const pagesBasePath = "/any-llm";' in not_found_redirect
    assert 'const docsBaseUrl = new URL("https://docs.mozilla.ai/any-llm/");' in not_found_redirect


def test_build_redirect_site_replaces_existing_output_and_skips_missing_passthroughs(tmp_path: Path) -> None:
    source_dir = tmp_path / "site"
    output_dir = tmp_path / "gh-pages-redirect"
    redirect_generator = _load_redirect_generator_module()

    _write_sample_site(source_dir, include_passthroughs=False, include_images=False)
    output_dir.mkdir(parents=True)
    (output_dir / "stale.txt").write_text("old\n", encoding="utf-8")

    redirect_count = redirect_generator.build_redirect_site(
        source_dir=source_dir,
        output_dir=output_dir,
        base_url="https://docs.mozilla.ai/any-llm",
        pages_base_path="any-llm",
    )

    assert redirect_count > 0
    assert not (output_dir / "stale.txt").exists()
    assert (output_dir / "index.html").exists()
    assert not (output_dir / "openapi.json").exists()
    assert not (output_dir / "llms.txt").exists()
    assert not (output_dir / "images").exists()


def test_parse_args_and_main_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = tmp_path / "site"
    output_dir = tmp_path / "custom-output"
    redirect_generator = _load_redirect_generator_module()

    _write_sample_site(source_dir, include_images=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_github_pages_redirects.py",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
            "--base-url",
            "https://docs.mozilla.ai/any-llm",
            "--pages-base-path",
            "any-llm",
        ],
    )

    args = redirect_generator.parse_args()

    assert args.source_dir == source_dir
    assert args.output_dir == output_dir
    assert args.base_url == "https://docs.mozilla.ai/any-llm"
    assert args.pages_base_path == "any-llm"

    exit_code = redirect_generator.main()
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "Done - " in stdout
    assert (output_dir / "404.html").exists()


def test_main_returns_error_for_missing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_source = tmp_path / "missing-site"
    output_dir = tmp_path / "gh-pages-redirect"
    redirect_generator = _load_redirect_generator_module()

    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_github_pages_redirects.py",
            "--source-dir",
            str(missing_source),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = redirect_generator.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert f"Input directory does not exist: {missing_source}" in captured.err


def test_script_entrypoint_exits_with_main_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "site"
    output_dir = tmp_path / "entrypoint-output"

    _write_sample_site(source_dir, include_images=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT_PATH),
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert (output_dir / "index.html").exists()
