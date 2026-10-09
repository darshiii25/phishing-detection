"""Character-level URL encoding module.

Maps raw URL strings into fixed-length integer ID sequences suitable for
Keras Embedding layers and 1D Convolutional Neural Networks.

Key Specifications:
- Reserved token ID 0: <PAD> (padding token)
- Reserved token ID 1: <UNK> (out-of-vocabulary unknown token)
- Distinct IDs >= 2: Individual ASCII characters
- Input shape:  List[str] or pd.Series of length N
- Output shape: np.ndarray of shape (N, max_url_len) with dtype np.int32
- Vocabulary serialization: JSON format for reproducible inference
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


# Default standard URL character set (86 characters identified from Dataset.csv)
DEFAULT_URL_CHARS: str = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " -_.~:/?#[]@!$&'()*+,;=%"
)


class UrlCharacterEncoder:
    """Character-level tokenizer and sequence encoder for URL strings."""

    def __init__(
        self,
        max_url_len: int = 200,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        """Initializes the character encoder.

        Args:
            max_url_len: Maximum sequence length (shorter URLs are padded, longer truncated).
            pad_token: Representation for padding (assigned ID 0).
            unk_token: Representation for unseen characters (assigned ID 1).
        """
        self.max_url_len = max_url_len
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Reserved indices
        self.pad_id = 0
        self.unk_id = 1

        self.char_to_id: Dict[str, int] = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }
        self.id_to_char: Dict[int, str] = {
            self.pad_id: self.pad_token,
            self.unk_id: self.unk_token,
        }
        self._is_fitted = False

    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size including reserved tokens."""
        return len(self.char_to_id)

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the encoder has a fitted vocabulary."""
        return self._is_fitted

    def fit(self, texts: Iterable[str], min_freq: int = 1) -> "UrlCharacterEncoder":
        """Fits vocabulary from a training collection of URL strings.

        Note: To prevent data leakage, fit ONLY on training data, never test data.

        Args:
            texts: Iterable of URL strings.
            min_freq: Minimum character frequency threshold to include in vocabulary.

        Returns:
            self
        """
        char_counts: Dict[str, int] = {}
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            for ch in text:
                char_counts[ch] = char_counts.get(ch, 0) + 1

        # Sort characters deterministically (by frequency descending, then alphabetically)
        sorted_chars = sorted(
            [ch for ch, count in char_counts.items() if count >= min_freq],
            key=lambda c: (-char_counts[c], c),
        )

        # Build mappings
        self.char_to_id = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }
        for ch in sorted_chars:
            if ch not in self.char_to_id:
                new_id = len(self.char_to_id)
                self.char_to_id[ch] = new_id

        self.id_to_char = {idx: ch for ch, idx in self.char_to_id.items()}
        self._is_fitted = True
        return self

    def fit_from_charset(self, charset: str = DEFAULT_URL_CHARS) -> "UrlCharacterEncoder":
        """Initializes vocabulary directly from a predefined character set.

        Provides fixed, deterministic character mapping independent of data splits.

        Args:
            charset: String containing all characters to include in vocabulary.

        Returns:
            self
        """
        self.char_to_id = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
        }
        # Sort for complete reproducibility
        for ch in sorted(set(charset)):
            if ch not in self.char_to_id:
                new_id = len(self.char_to_id)
                self.char_to_id[ch] = new_id

        self.id_to_char = {idx: ch for ch, idx in self.char_to_id.items()}
        self._is_fitted = True
        return self

    def encode_url(self, url: str) -> np.ndarray:
        """Encodes a single URL string into a 1D integer array.

        Args:
            url: Input URL string.

        Returns:
            1D np.ndarray of shape (max_url_len,) with dtype np.int32.
        """
        if not self._is_fitted:
            raise RuntimeError("UrlCharacterEncoder must be fitted before encoding.")

        if not isinstance(url, str):
            url = str(url)

        seq = [self.char_to_id.get(ch, self.unk_id) for ch in url]

        # Truncate if exceeds max_url_len
        if len(seq) > self.max_url_len:
            seq = seq[: self.max_url_len]
        # Pad with pad_id (0) at the end if shorter
        elif len(seq) < self.max_url_len:
            seq = seq + [self.pad_id] * (self.max_url_len - len(seq))

        return np.array(seq, dtype=np.int32)

    def encode_batch(
        self, urls: Union[Sequence[str], pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Encodes a batch/collection of URLs into a 2D integer matrix.

        Args:
            urls: Collection of URL strings (length N).

        Returns:
            2D np.ndarray of shape (N, max_url_len) with dtype np.int32.
        """
        if not self._is_fitted:
            raise RuntimeError("UrlCharacterEncoder must be fitted before encoding.")

        n_samples = len(urls)
        matrix = np.full((n_samples, self.max_url_len), self.pad_id, dtype=np.int32)

        for i, url in enumerate(urls):
            if not isinstance(url, str):
                url = str(url)
            length = min(len(url), self.max_url_len)
            for j in range(length):
                matrix[i, j] = self.char_to_id.get(url[j], self.unk_id)

        return matrix

    def decode_sequence(self, sequence: Sequence[int]) -> str:
        """Decodes an integer sequence back to a readable string (stripping padding).

        Args:
            sequence: Integer IDs.

        Returns:
            Reconstructed URL string.
        """
        chars = []
        for idx in sequence:
            if idx == self.pad_id:
                continue
            chars.append(self.id_to_char.get(int(idx), "?"))
        return "".join(chars)

    def save_vocabulary(self, file_path: Union[str, Path]) -> None:
        """Serializes vocabulary and configuration to a JSON file.

        Args:
            file_path: Output file path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_url_len": self.max_url_len,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "char_to_id": self.char_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_vocabulary(cls, file_path: Union[str, Path]) -> "UrlCharacterEncoder":
        """Deserializes vocabulary from a JSON file.

        Args:
            file_path: Path to serialized vocabulary JSON.

        Returns:
            Instantiated and fitted UrlCharacterEncoder.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encoder = cls(
            max_url_len=data["max_url_len"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
        )
        encoder.pad_id = data["pad_id"]
        encoder.unk_id = data["unk_id"]
        encoder.char_to_id = data["char_to_id"]
        encoder.id_to_char = {int(idx): ch for ch, idx in data["char_to_id"].items()}
        encoder._is_fitted = True
        return encoder
