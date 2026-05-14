import numpy as np

from driftscope.embeddings import MockEmbedder


def test_mock_embedder_is_deterministic() -> None:
    embedder = MockEmbedder(dim=32)
    vectors_a = embedder.embed(["teal blazer", "skylark rewards"])
    vectors_b = embedder.embed(["teal blazer", "skylark rewards"])

    assert np.allclose(vectors_a, vectors_b)
    assert vectors_a.shape == (2, 32)
    norms = np.linalg.norm(vectors_a, axis=1)
    assert np.allclose(norms, 1.0)


def test_mock_embedder_similar_texts_have_nonzero_cosine() -> None:
    embedder = MockEmbedder(dim=128)
    shared = embedder.embed([
        "teal blazer and brooch for aunt retirement",
        "teal blazer gift for aunt",
    ])
    disjoint = embedder.embed([
        "skylark rewards free nights",
        "backyard compost bin setup",
    ])

    shared_cos = float(np.dot(shared[0], shared[1]))
    disjoint_cos = float(np.dot(disjoint[0], disjoint[1]))

    assert shared_cos > disjoint_cos
    assert shared_cos > 0.0
