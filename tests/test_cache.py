"""Tests for the content-addressed analysis cache."""

from pathlib import Path

from controlnet_mcp.cache import AnalysisCache, image_digest

PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def test_digest_is_stable_and_short() -> None:
    digest = image_digest(b"reference bytes")

    assert digest == image_digest(b"reference bytes")
    assert len(digest) == 16
    assert digest == digest.lower()
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest != image_digest(b"other bytes")


def test_path_layout(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)

    assert cache.path_for("abc123", "depth", "1", 512) == (tmp_path / "abc123" / "depth-v1-512.png")


def test_path_layout_with_variant(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)

    assert cache.path_for("abc", "segments", "1", 512, "abcd1234") == (
        tmp_path / "abc" / "segments-v1-512-abcd1234.png"
    )


def test_get_returns_none_when_absent(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)

    assert cache.get("abc123", "depth", "1", 512) is None


def test_put_then_get_roundtrip(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    png = PNG_HEADER + b"rendered depth map"

    stored = cache.put("abc123", "depth", "1", 512, png)

    assert stored == tmp_path / "abc123" / "depth-v1-512.png"
    assert stored.is_file()
    assert cache.get("abc123", "depth", "1", 512) == png
    assert [entry.name for entry in stored.parent.iterdir()] == ["depth-v1-512.png"]
