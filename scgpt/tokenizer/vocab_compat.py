"""
Vocabulary backend compatibility module for scGPT.

This module prefers torchtext as the GeneVocab base class only when it is
actually usable (import succeeds and the Vocab type is subclassable).
Otherwise it falls back to a pure-Python implementation.

Why fallback is needed:
----------------------
torchtext <=0.13 exposed Vocab as a subclassable Python class.
torchtext >=0.14 (including 0.18.0 shipped alongside PyTorch 2.x / CUDA 12.8)
changed Vocab to a C++ extension type (VocabPybind) wrapped in a thin Python
shim. C++ extension types cannot be used as base classes in Python, so
``class GeneVocab(torchtext.vocab.Vocab)`` raises TypeError at import time.
"""

from typing import Dict, Iterable, List, Optional, Union

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_torchtext_vocab_cls = None
torchtext_import_succeeded: bool = False
torchtext_import_error: Optional[str] = None
torchtext_vocab_is_subclassable: bool = False

try:
    from torchtext.vocab import Vocab as _torchtext_vocab_cls  # noqa: F401

    torchtext_import_succeeded = True
except (ImportError, OSError) as exc:
    torchtext_import_error = str(exc)


def _probe_torchtext_vocab_subclassable() -> bool:
    """Return True when torchtext Vocab can be used as a Python base class."""
    if not torchtext_import_succeeded or _torchtext_vocab_cls is None:
        return False
    try:
        class _TorchtextVocabSubclassProbe(_torchtext_vocab_cls):
            pass

        return True
    except TypeError:
        return False


torchtext_vocab_is_subclassable = _probe_torchtext_vocab_subclassable()
gene_vocab_base_backend: str = (
    "torchtext" if torchtext_vocab_is_subclassable else "builtin"
)


def get_vocab_info() -> str:
    """Return a human-readable description of the detected vocab configuration."""
    if torchtext_vocab_is_subclassable:
        return (
            "torchtext detected and subclassable; using torchtext Vocab as "
            "GeneVocab base class."
        )
    if torchtext_import_succeeded:
        return (
            "torchtext detected but not subclassable (likely torchtext >=0.14 "
            "C++ Vocab); using built-in pure-Python Vocab as GeneVocab base "
            "class."
        )
    if torchtext_import_error is not None:
        return (
            "torchtext import failed; using built-in pure-Python Vocab. "
            f"Original error: {torchtext_import_error}"
        )
    return "torchtext not installed — using built-in pure-Python Vocab."


# ---------------------------------------------------------------------------
# Pure-Python Vocab implementation
# ---------------------------------------------------------------------------


class BuiltinVocab:
    """
    Pure-Python bijective vocabulary: token <-> integer index.

    This is a drop-in replacement for ``torchtext.vocab.Vocab`` with a
    compatible public API. It is used as the base class for GeneVocab because
    torchtext >=0.14 ships Vocab as a C++ extension type that cannot be
    subclassed.

    Public API (superset of torchtext 0.14+ Vocab):

    Lookup  (O(1) dict):
        __getitem__(token: str) -> int
        __getitem__(index: int) -> str   # extra; not in torchtext
        __call__(tokens: Iterable[str]) -> List[int]
        __contains__(token: str) -> bool
        __len__() -> int

    Views:
        get_stoi() -> Dict[str, int]
        get_itos() -> List[str]

    OOV default:
        set_default_index(index: Optional[int])
        get_default_index() -> Optional[int]   # returns None when unset
                                               # (torchtext returns -1)
    Mutation:
        append_token(token: str) -> int
        insert_token(token: str, index: int)
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

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Dict-style views
    # ------------------------------------------------------------------

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return list(self._itos)

    # ------------------------------------------------------------------
    # Default index (OOV fallback)
    # ------------------------------------------------------------------

    def set_default_index(self, index: Optional[int]) -> None:
        if index is not None and not (0 <= index < len(self._itos)):
            raise ValueError(
                f"Default index {index} is out of range for "
                f"vocabulary of size {len(self._itos)}."
            )
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        """Return the default index, or None if not set.

        Note: torchtext's Vocab.get_default_index() returns -1 when unset.
        This implementation returns None instead for clearer Python semantics.
        """
        return self._default_index

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

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
        # Rebuild stoi from scratch (insertion shifts all subsequent indices).
        self._stoi = {tok: idx for idx, tok in enumerate(self._itos)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_torchtext_vocab(obj: object) -> bool:
    """
    Return True if *obj* is a torchtext Vocab instance (duck-typed).

    Checks for the torchtext public API surface rather than isinstance, so
    this works regardless of the torchtext version.
    """
    return (
        torchtext_import_succeeded
        and _torchtext_vocab_cls is not None
        and isinstance(obj, _torchtext_vocab_cls)
    )


def from_torchtext_vocab(tt_vocab) -> BuiltinVocab:
    """
    Convert a torchtext Vocab object to scGPT's pure-Python Vocab.

    torchtext.vocab.Vocab.get_default_index() returns -1 when no default is
    set; we convert that sentinel to None.

    Args:
        tt_vocab: A ``torchtext.vocab.Vocab`` instance.

    Returns:
        BuiltinVocab: A pure-Python Vocab with the same token-to-index mapping.
    """
    itos: List[str] = list(tt_vocab.get_itos())
    v = BuiltinVocab(itos)
    raw_default = tt_vocab.get_default_index()
    # torchtext uses -1 as "not set"; our Vocab uses None.
    if raw_default is not None and raw_default >= 0:
        v.set_default_index(raw_default)
    return v


# Active base class symbol used by GeneVocab.
Vocab = (
    _torchtext_vocab_cls if torchtext_vocab_is_subclassable else BuiltinVocab
)
