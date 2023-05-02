from pathlib import Path
from scgpt.tokenizer import GeneVocab, get_default_gene_vocab


def test_gene_vocab():
    gene_vocab = GeneVocab([])
    assert len(gene_vocab) == 0

    gene_vocab.append_token("abc")
    assert len(gene_vocab) == 1

    gene_vocab = GeneVocab(gene_vocab)
    assert gene_vocab["abc"] == 0

    gene_vocab = GeneVocab(["def", "g", "h"], specials=["a", "b", "c"])
    assert gene_vocab["a"] == 0
    assert gene_vocab["def"] == 3
    assert len(gene_vocab) == 6

    gene_vocab = GeneVocab(["a"], specials=["<pad>"], special_first=False)
    assert gene_vocab["<pad>"] == 1
    assert gene_vocab.get_default_index() == 1


def test_gene_vocab_from_dict():
    gene_vocab = GeneVocab.from_dict({"a": 0, "b": 1, "c": 2})
    assert len(gene_vocab) == 3
    assert gene_vocab["a"] == 0
    assert gene_vocab["c"] == 2


def test_gene_vocab_from_file():
    # current file path
    test_file = Path(__file__).parent / "vocab.json"
    gene_vocab = GeneVocab.from_file(test_file)
    assert len(gene_vocab) == 3


def test_gene_vocab_pad_token():
    gene_vocab = GeneVocab(["a", "b", "c"], specials=["<pad>"])
    assert gene_vocab.pad_token is None
    gene_vocab.pad_token = "<pad>"
    assert gene_vocab.pad_token == "<pad>"


def test_get_default_gene_vocab():
    gene_vocab = get_default_gene_vocab()
    assert gene_vocab["A12M1"] == 0
