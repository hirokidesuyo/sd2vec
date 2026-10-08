import numpy as np

from sd2vec.vectors import SDVectors


def test_most_similar_excludes_query():
    vectors = SDVectors(
        ["猫", "犬", "戦争"],
        ["pleasant", "safe"],
        np.array([[1, 1], [0.9, 0.9], [-1, -1]], dtype=np.float32),
    )
    result = vectors.most_similar("猫", topn=2)
    assert result[0][0] == "犬"
    assert all(name != "猫" for name, _ in result)


def test_vector_size():
    vectors = SDVectors(["猫"], ["pleasant", "safe"], np.zeros((1, 2)))
    assert vectors.vector_size == 2
