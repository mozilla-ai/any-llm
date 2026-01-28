"""Unit tests for the platform provider."""

import base64

import pytest

from any_llm.providers.platform import (
    ANY_LLM_KEY_PATTERN,
    DecryptionError,
    InvalidKeyFormatError,
    KeyComponents,
    decrypt_sealed_box,
    is_platform_key,
    load_private_key,
    parse_any_llm_key,
)


class TestParseAnyLlmKey:
    """Tests for parse_any_llm_key function."""

    def test_parse_valid_key(self) -> None:
        """Test parsing a valid ANY_LLM_KEY."""
        key = "ANY.v1.abc123.xyz789-SGVsbG9Xb3JsZA=="
        result = parse_any_llm_key(key)

        assert isinstance(result, KeyComponents)
        assert result.key_id == "abc123"
        assert result.public_key_fingerprint == "xyz789"
        assert result.base64_encoded_private_key == "SGVsbG9Xb3JsZA=="

    def test_parse_key_with_special_characters(self) -> None:
        """Test parsing key with special characters in fingerprint."""
        key = "ANY.v1.key-id-123.fp_456-SGVsbG8gV29ybGQh"
        result = parse_any_llm_key(key)

        assert result.key_id == "key-id-123"
        assert result.public_key_fingerprint == "fp_456"
        assert result.base64_encoded_private_key == "SGVsbG8gV29ybGQh"

    def test_parse_invalid_key_missing_prefix(self) -> None:
        """Test parsing key with missing ANY prefix."""
        with pytest.raises(InvalidKeyFormatError):
            parse_any_llm_key("v1.abc123.xyz789-SGVsbG9Xb3JsZA==")

    def test_parse_invalid_key_wrong_version(self) -> None:
        """Test parsing key with wrong version."""
        with pytest.raises(InvalidKeyFormatError):
            parse_any_llm_key("ANY.v2.abc123.xyz789-SGVsbG9Xb3JsZA==")

    def test_parse_invalid_key_empty(self) -> None:
        """Test parsing empty key."""
        with pytest.raises(InvalidKeyFormatError):
            parse_any_llm_key("")

    def test_parse_invalid_key_wrong_format(self) -> None:
        """Test parsing key with wrong format."""
        with pytest.raises(InvalidKeyFormatError):
            parse_any_llm_key("INVALID_KEY")


class TestLoadPrivateKey:
    """Tests for load_private_key function."""

    def test_load_valid_32_byte_key(self) -> None:
        """Test loading a valid 32-byte private key."""
        # Create a valid 32-byte key
        key_bytes = b"0" * 32
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")

        result = load_private_key(key_b64)

        assert result == key_bytes
        assert len(result) == 32

    def test_load_key_wrong_size_too_short(self) -> None:
        """Test loading a private key that's too short."""
        key_bytes = b"short"
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")

        with pytest.raises(InvalidKeyFormatError):
            load_private_key(key_b64)

    def test_load_key_wrong_size_too_long(self) -> None:
        """Test loading a private key that's too long."""
        key_bytes = b"0" * 64
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")

        with pytest.raises(InvalidKeyFormatError):
            load_private_key(key_b64)

    def test_load_key_invalid_base64(self) -> None:
        """Test loading a key with invalid base64 encoding."""
        with pytest.raises(InvalidKeyFormatError):
            load_private_key("not-valid-base64!!!")


class TestIsPlatformKey:
    """Tests for is_platform_key function."""

    def test_valid_platform_key(self) -> None:
        """Test that a valid platform key is recognized."""
        assert is_platform_key("ANY.v1.abc123.xyz789-SGVsbG9Xb3JsZA==") is True

    def test_invalid_platform_key(self) -> None:
        """Test that an invalid key is not recognized as platform key."""
        assert is_platform_key("sk-1234567890") is False

    def test_none_key(self) -> None:
        """Test that None is not recognized as platform key."""
        assert is_platform_key(None) is False

    def test_empty_key(self) -> None:
        """Test that empty string is not recognized as platform key."""
        assert is_platform_key("") is False


class TestKeyPattern:
    """Tests for the ANY_LLM_KEY_PATTERN regex."""

    def test_pattern_matches_valid_key(self) -> None:
        """Test that pattern matches valid key format."""
        key = "ANY.v1.kid123.fp456-YmFzZTY0a2V5"
        match = ANY_LLM_KEY_PATTERN.match(key)

        assert match is not None
        groups = match.groups()
        assert groups == ("kid123", "fp456", "YmFzZTY0a2V5")

    def test_pattern_does_not_match_invalid_version(self) -> None:
        """Test that pattern does not match invalid version."""
        assert ANY_LLM_KEY_PATTERN.match("ANY.v2.kid.fp-key") is None

    def test_pattern_does_not_match_missing_components(self) -> None:
        """Test that pattern does not match key with missing components."""
        assert ANY_LLM_KEY_PATTERN.match("ANY.v1.kid-key") is None
        assert ANY_LLM_KEY_PATTERN.match("ANY.v1.kid.fp") is None


class TestDecryptSealedBox:
    """Tests for decrypt_sealed_box function."""

    def test_decrypt_sealed_box_too_short(self) -> None:
        """Test decrypting sealed box that's too short raises DecryptionError."""
        # Create a 32-byte private key
        private_key = b"0" * 32

        # Create data that's less than 48 bytes (minimum for sealed box)
        short_data = b"0" * 47
        short_data_b64 = base64.b64encode(short_data).decode("utf-8")

        with pytest.raises(DecryptionError):
            decrypt_sealed_box(short_data_b64, private_key)

    def test_decrypt_sealed_box_invalid_base64(self) -> None:
        """Test decrypting sealed box with invalid base64 raises DecryptionError."""
        private_key = b"0" * 32

        with pytest.raises(DecryptionError):
            decrypt_sealed_box("not-valid-base64!!!", private_key)
