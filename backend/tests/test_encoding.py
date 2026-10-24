"""Unit tests for character-level URL encoding module."""

from pathlib import Path
import numpy as np
import pytest

from src.encoding import UrlCharacterEncoder, DEFAULT_URL_CHARS


def test_encoder_not_fitted_error():
    """Verifies that calling encode before fitting raises RuntimeError."""
    encoder = UrlCharacterEncoder(max_url_len=50)
    with pytest.raises(RuntimeError):
        encoder.encode_url("http://example.com")


def test_fit_and_encode_url_shape_and_padding():
    """Verifies 1D encoded array shape, integer dtype, and padding correctness."""
    encoder = UrlCharacterEncoder(max_url_len=20)
    train_urls = ["http://abc.com", "https://xyz.org"]
    encoder.fit(train_urls)

    encoded = encoder.encode_url("http://abc.com")
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (20,)
    assert encoded.dtype == np.int32

    # String length is 14; positions 0..13 should be > 0, 14..19 should be pad_id (0)
    url_len = len("http://abc.com")
    assert (encoded[:url_len] >= 2).all()
    assert (encoded[url_len:] == encoder.pad_id).all()


def test_truncation_when_exceeding_max_len():
    """Verifies truncation when URL length exceeds max_url_len."""
    max_len = 10
    encoder = UrlCharacterEncoder(max_url_len=max_len)
    encoder.fit(["http://longurl.example.com"])

    encoded = encoder.encode_url("http://longurl.example.com")
    assert len(encoded) == max_len


def test_unknown_character_mapped_to_unk():
    """Verifies unseen characters map to unk_id without crashing."""
    encoder = UrlCharacterEncoder(max_url_len=15)
    # Fit only on lowercase 'a'..'z' and '.'
    encoder.fit(["abc.com"])

    # Encode string containing unseen uppercase, digits, and unicode/special character
    encoded = encoder.encode_url("a@b#1")
    # 'a' is known (>= 2)
    # '@' is unknown -> unk_id (1)
    # 'b' is known (>= 2)
    # '#' is unknown -> unk_id (1)
    # '1' is unknown -> unk_id (1)
    assert encoded[0] >= 2
    assert encoded[1] == encoder.unk_id
    assert encoded[2] >= 2
    assert encoded[3] == encoder.unk_id
    assert encoded[4] == encoder.unk_id


def test_encode_batch_shape():
    """Verifies 2D batch output dimensions and values."""
    encoder = UrlCharacterEncoder(max_url_len=30)
    urls = ["http://a.com", "http://b.com", "http://c.org/path"]
    encoder.fit(urls)

    batch_matrix = encoder.encode_batch(urls)
    assert batch_matrix.shape == (3, 30)
    assert batch_matrix.dtype == np.int32


def test_save_and_load_vocabulary(tmp_path: Path):
    """Verifies vocabulary serialization and reload round-trip consistency."""
    encoder = UrlCharacterEncoder(max_url_len=25)
    sample_urls = ["http://alpha.com", "http://beta.org/test"]
    encoder.fit(sample_urls)

    test_url = "http://alpha.com/test"
    orig_encoded = encoder.encode_url(test_url)

    save_file = tmp_path / "vocab_test.json"
    encoder.save_vocabulary(save_file)
    assert save_file.is_file()

    loaded_encoder = UrlCharacterEncoder.load_vocabulary(save_file)
    assert loaded_encoder.max_url_len == 25
    assert loaded_encoder.vocab_size == encoder.vocab_size
    assert loaded_encoder.char_to_id == encoder.char_to_id

    reloaded_encoded = loaded_encoder.encode_url(test_url)
    np.testing.assert_array_equal(orig_encoded, reloaded_encoded)


def test_fit_from_charset():
    """Verifies deterministic initialization from character set."""
    encoder = UrlCharacterEncoder(max_url_len=50)
    encoder.fit_from_charset(DEFAULT_URL_CHARS)

    assert encoder.is_fitted
    assert encoder.vocab_size == len(set(DEFAULT_URL_CHARS)) + 2  # + PAD + UNK
    assert encoder.char_to_id["a"] >= 2
    assert encoder.char_to_id[encoder.pad_token] == 0
    assert encoder.char_to_id[encoder.unk_token] == 1


def test_decode_sequence():
    """Verifies reconstructing string from sequence of integer IDs."""
    encoder = UrlCharacterEncoder(max_url_len=10)
    encoder.fit(["test"])
    seq = encoder.encode_url("test")
    decoded = encoder.decode_sequence(seq)
    assert decoded == "test"


def test_consistent_vocabulary_across_splits():
    """Verifies that vocabulary fit on train partition encodes val and test splits consistently."""
    train_urls = ["http://bank-login.com", "http://legit-site.org"]
    val_urls = ["http://bank-login.com/account", "http://new-domain.io"]
    test_urls = ["https://secure-bank-login.com", "http://unknown-chars-^%$.net"]

    encoder = UrlCharacterEncoder(max_url_len=30)
    # Fit strictly on train partition
    encoder.fit(train_urls)
    vocab_size_before = encoder.vocab_size

    # Encode train, val, test
    X_train_enc = encoder.encode_batch(train_urls)
    X_val_enc = encoder.encode_batch(val_urls)
    X_test_enc = encoder.encode_batch(test_urls)

    # Vocabulary must remain unchanged (no leak from val/test)
    assert encoder.vocab_size == vocab_size_before

    # Verify output shapes and types
    assert X_train_enc.shape == (2, 30)
    assert X_val_enc.shape == (2, 30)
    assert X_test_enc.shape == (2, 30)
    assert X_train_enc.dtype == np.int32
    assert X_val_enc.dtype == np.int32
    assert X_test_enc.dtype == np.int32

    # Verify ID range across all splits
    for arr in [X_train_enc, X_val_enc, X_test_enc]:
        assert (arr >= 0).all()
        assert (arr < encoder.vocab_size).all()

    # Verify consistent character encoding: 'b' in train must match 'b' in val and test
    b_id = encoder.char_to_id["b"]
    assert X_train_enc[0, 7] == b_id   # "http://" is 7 chars, index 7 is 'b'
    assert X_val_enc[0, 7] == b_id     # "http://" is 7 chars, index 7 is 'b'

