from mabwiser.mab import LearningPolicy
from pytest import approx, mark, raises

from backend.llm.mab_ranker import MultiArmedBanditRanker
from backend.llm.ranking import (
    CHOIX_RANKER_ALGORITHMS,
    ChoixRanker,
    ChoixRankerConfIntervals,
    EloRanker,
    PerCategoryRanker,
    _elo_to_probabilities,
    _score_to_elo,
)
from backend.llm.utils import AnnotatedFloat
from db.ratings import OVERALL_CATEGORY_NAME


def _assert_approx(annotated_float: AnnotatedFloat, value: float | None, annotation: str) -> None:
    assert value == approx(annotated_float.value, abs=0.001)
    assert annotation == annotated_float.annotation


def test_elo_ranker() -> None:
    ranker = EloRanker()

    assert ranker.get_annotated_rating("a") == AnnotatedFloat(
        1000, "Starting value 1000.0; 0 adjustments (0 negative, mean=0.00, stdev=0.00)"
    )
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(0.5, "rating_a=1000.00, rating_b=1000.00")

    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)

    _assert_approx(
        ranker.get_annotated_rating("a"),
        1005.931,
        "Starting value 1000.0; 3 adjustments (0 negative, mean=1.98, stdev=0.02)",
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.517, "rating_a=1005.93, rating_b=994.07")

    ranker.update("a", "b", 0.0)
    ranker.update("a", "b", 0.0)

    _assert_approx(
        ranker.get_annotated_rating("a"),
        1001.818,
        "Starting value 1000.0; 5 adjustments (2 negative, mean=0.36, stdev=1.98)",
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.505, "rating_a=1001.82, rating_b=998.18")

    for _ in range(1000):
        ranker.update("a", "b", 0.0)
    _assert_approx(
        ranker.get_annotated_rating("a"),
        672.676,
        "Starting value 1000.0; 1005 adjustments (1002 negative, mean=-0.33, stdev=0.39)",
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.023, "rating_a=672.68, rating_b=1327.32")


@mark.filterwarnings("ignore:Mean of empty slice")
@mark.filterwarnings("ignore:invalid value encountered in scalar divide")
@mark.parametrize("algo", CHOIX_RANKER_ALGORITHMS)
def test_choix_ranker(algo: str) -> None:
    ranker = ChoixRanker(choix_ranker_algorithm=algo)

    assert ranker.get_annotated_rating("a") == AnnotatedFloat(None, "Wins: 0, Losses: 0, Ties: 0")
    with raises(ValueError):
        ranker.predict("a", "b")

    for _ in range(10):
        ranker.update("a", "b", 1.0)

    assert ranker.get_rating("a") > 3.5  # type: ignore
    assert ranker.get_rating("b") < -3.5  # type: ignore

    for _ in range(10):
        ranker.update("a", "b", 1.0)
    for _ in range(10):
        ranker.update("a", "b", 0.0)
    for _ in range(10):
        ranker.update("a", "b", 0.5)

    _assert_approx(ranker.get_annotated_rating("a"), 1081.093, "Wins: 20, Losses: 10, Ties: 10")
    _assert_approx(ranker.get_annotated_rating("b"), 918.906, "Wins: 10, Losses: 20, Ties: 10")
    _assert_approx(ranker.predict_annotate("a", "b"), 0.6, "rating_a=1081, rating_b=919")

    for _ in range(5):
        ranker.update("a", "b", 0.0)

    leaderboard = ranker.leaderboard()
    assert len(leaderboard) == 2
    assert leaderboard[0].model == "a"
    assert leaderboard[1].model == "b"
    assert leaderboard[0].rating == approx(1036.464, abs=0.01)
    assert leaderboard[1].rating == approx(963.535, abs=0.01)


def test_mab_ranker() -> None:
    ranker = MultiArmedBanditRanker(
        models=["a", "b"],
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
    )
    ranker.fit([("a", "b"), ("a", "b"), ("a", "b"), ("a", "b")], [1.0, 0.0, 1.0, 1.0])

    assert ranker.get_rating("a") == 0.75
    with raises(NotImplementedError):
        ranker.predict("a", "b")


def test_choix_confidence_ranker() -> None:
    non_ci_ranker = ChoixRanker(models=["a", "b"], choix_ranker_algorithm="lsr_pairwise")
    ci_ranker = ChoixRankerConfIntervals(
        models=["a", "b"], choix_ranker_algorithm="lsr_pairwise", num_bootstrap_iterations=10
    )

    # The confidence intervals should get wider as data is more spread out.
    for _ in range(10):
        ci_ranker.update("a", "b", 1.0)
        non_ci_ranker.update("a", "b", 1.0)
    rating, lower, upper = ci_ranker.get_rating_conf_intervals("a")
    assert lower == rating == upper == approx(2703.4786)  # no ambiguity

    for _ in range(5):
        ci_ranker.update("a", "b", 0.0)
        non_ci_ranker.update("a", "b", 0.0)
    rating, lower, upper = ci_ranker.get_rating_conf_intervals("a")
    assert lower == approx(1077.0497, abs=0.001)
    assert rating == approx(1202.256, abs=0.001)
    assert upper == approx(1374.191, abs=0.001)

    # ... and narrower as data is more consistent.
    for _ in range(50):
        ci_ranker.update("a", "b", 0.0)
        non_ci_ranker.update("a", "b", 0.0)
    rating, lower, upper = ci_ranker.get_rating_conf_intervals("a")
    assert lower == approx(602.914, abs=0.001)
    assert rating == approx(659.083, abs=0.001)
    assert upper == approx(713.847, abs=0.001)

    # Mean rank should be similar to the rank from a non-confidence-intervals ranker.
    assert ci_ranker.get_rating("a") == approx(non_ci_ranker.get_rating("a"), rel=0.03)
    assert ci_ranker.get_rating("b") == approx(non_ci_ranker.get_rating("b"), rel=0.03)


def test_per_category_ranker() -> None:
    ranker_kwargs = dict(
        models=["a", "b"],
        choix_ranker_algorithm="ilsr_pairwise",
        num_bootstrap_iterations=10,
    )
    categories = ("coding", "math")
    ranker = PerCategoryRanker(categories=categories, ranker_cls=ChoixRankerConfIntervals, ranker_kwargs=ranker_kwargs)
    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0, category="coding")
    ranker.update("b", "a", 0.0, category="coding")
    ranker.update("a", "b", 0.0, category="math")
    ranker.update("b", "a", 1.0, category="math")

    # Overall, a is slightly better than b.
    expected_overall_ratings = {"a": approx(1081.026), "b": approx(918.973)}
    assert ranker.get_ratings() == expected_overall_ratings
    assert ranker.get_ratings_all_categories() == {
        # In coding, a is much better than b.
        "coding": {"a": approx(2381.751), "b": approx(-381.751)},
        # In math, b is much better than a.
        "math": {"b": approx(2381.751), "a": approx(-381.751)},
        OVERALL_CATEGORY_NAME: expected_overall_ratings,
    }

    with raises(ValueError):
        ranker.update("a", "b", 1.0, category="non-existent category")

    rating_all_categories = ranker.get_annotated_rating_all_categories("a")

    assert len(rating_all_categories) == 3
    _assert_approx(rating_all_categories["coding"], 2381.751, "Wins: 2, Losses: 0, Ties: 0 (2381.8 to 2381.8)")
    _assert_approx(rating_all_categories["math"], -381.751, "Wins: 0, Losses: 2, Ties: 0 (-381.8 to -381.8)")
    _assert_approx(
        rating_all_categories[OVERALL_CATEGORY_NAME], 1081.026, "Wins: 3, Losses: 2, Ties: 0 (811.2 to 2564.9)"
    )

    assert ranker.annotate_prediction_all_categories("a", "b") == {
        "coding": "rating_a=2382, rating_b=-382",
        "math": "rating_a=-382, rating_b=2382",
        OVERALL_CATEGORY_NAME: "rating_a=1081, rating_b=919",
    }


def test_elo_to_probabilities() -> None:
    assert _elo_to_probabilities({"a": 1000, "b": 1000}) == {"a": 0.5, "b": 0.5}
    assert _elo_to_probabilities({"a": 800, "b": 1000}) == {"a": approx(0.25, abs=0.01), "b": approx(0.75, abs=0.01)}


def test_score_to_elo() -> None:
    assert _score_to_elo(0.1) == 1040
    assert _score_to_elo(0.5) == 1200
    assert _score_to_elo(0.5, init_rating=100) == 300
