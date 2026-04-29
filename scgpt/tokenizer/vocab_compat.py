"""
Pure-Python vocabulary for scGPT.

Replaces the former torchtext.vocab.Vocab dependency with a self-contained
implementation. torchtext is no longer required.
"""

from typing import Dict, Iterable, List, Optional, Union


class Vocab:
    """
    Pure-Python bijective vocabulary: token <-> integer index.

    Drop-in replacement for the former ``torchtext.vocab.Vocab`` base class.
    """

    def __init__(
        self,
        tokens: Optional[Iterable[str]] = None,
        default_index: Optional[int] = None,
    ) -> None:
        self._itos: List[str] = []
        self._stoi: Dict[str, int] = {}
        self._default_index: Optional[int] = default_index

        if tokens is not None:
            for token in tokens:
                self.append_token(token)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return len(self._itos)

    def __getitem__(self, item: Union[str, int]) -> Union[int, str]:
        if isinstance(item, int):
            return self._itos[item]
        if item in self._stoi:
            return self._stoi[item]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token {item!r} is not in the vocabulary.")

    def __call__(self, tokens: Iterable[str]) -> List[int]:
        return [self[token] for token in tokens]

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return list(self._itos)

    def set_default_index(self, index: Optional[int]) -> None:
        if index is not None and not (0 <= index < len(self._itos)):
            raise ValueError(
                f"Default index {index} is out of range for "
                f"vocabulary of size {len(self._itos)}."
            )
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    def append_token(self, token: str) -> int:
        """Append a token and return its index. No-op if already present."""
        if token in self._stoi:
            return self._stoi[token]
        index = len(self._itos)
        self._itos.append(token)
        self._stoi[token] = index
        return index

    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index, shifting later tokens up."""
        if index < 0 or index > len(self._itos):
            raise IndexError(
                f"Index {index} is out of range for "
                f"vocabulary of size {len(self._itos)}."
            )
        if token in self._stoi:
            if self._stoi[token] != index:
                raise ValueError(
                    f"Token {token!r} already exists at index "
                    f"{self._stoi[token]}, cannot re-insert at {index}."
                )
            return
        self._itos.insert(index, token)
        self._stoi = {tok: idx for idx, tok in enumerate(self._itos)}


def convert_legacy_vocab(obj) -> Vocab:
    """
    Convert a legacy vocab object (e.g. from an old torchtext pickle) to Vocab.

    Handles objects with `get_stoi()`, objects with `.vocab.get_stoi()`
    (torchtext wrapper), and plain dicts.
    """
    if isinstance(obj, Vocab):
        return obj
    if isinstance(obj, dict):
        tokens = [t for t, _ in sorted(obj.items(), key=lambda x: x[1])]
        return Vocab(tokens)
    if hasattr(obj, "get_stoi"):
        stoi = obj.get_stoi()
        tokens = [t for t, _ in sorted(stoi.items(), key=lambda x: x[1])]
        v = Vocab(tokens)
        raw_default = getattr(obj, "get_default_index", lambda: None)()
        if raw_default is not None and raw_default >= 0:
            v.set_default_index(raw_default)
        return v
    if hasattr(obj, "vocab") and hasattr(obj.vocab, "get_stoi"):
        return convert_legacy_vocab(obj.vocab)
    raise ValueError(
        f"Cannot convert {type(obj).__name__} to Vocab. "
        "Expected an object with get_stoi(), a .vocab attribute, or a dict."
    )
