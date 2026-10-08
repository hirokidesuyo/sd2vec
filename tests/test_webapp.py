import numpy as np

from webapp import explain_pair


def test_explain_pair_reports_shared_and_opposite_axes():
    concepts = ["左", "右", "中"]
    dimensions = [
        {"id": "pleasant", "left": "快い", "right": "不快な"},
        {"id": "strong", "left": "強い", "right": "弱い"},
        {"id": "active", "left": "活動的な", "right": "静的な"},
    ]
    vectors = np.array(
        [
            [1.0, 1.0, 0.2],
            [1.0, -1.0, 0.2],
            [0.0, 0.0, 0.0],
        ]
    )

    result = explain_pair("左", "右", concepts, dimensions, vectors)

    assert abs(result["cosine"] - 0.019608) < 1e-6
    assert result["opposite_axes"][0]["id"] == "strong"
    assert result["shared_axes"][0]["id"] == "pleasant"
    assert result["different_axes"][0]["id"] == "strong"


def test_explain_pair_rejects_unknown_words():
    dimensions = [{"id": "pleasant", "left": "快い", "right": "不快な"}]
    vectors = np.array([[1.0], [-1.0]])

    try:
        explain_pair("未知語", "右", ["左", "右"], dimensions, vectors)
    except KeyError as error:
        assert "未知語" in str(error)
    else:
        raise AssertionError("unknown words must raise KeyError")
