from mabwiser.mab import LearningPolicy
from pytest import approx, mark, raises

from backend.llm.mab_ranker import MultiArmedBanditRanker
from backend.llm.ranking import (
    CHOIX_RANKER_ALGORITHMS,
    ChoixRanker,
    ChoixRankerConfIntervals,
    EloRanker,
    NaiveRanker,
    PerCategoryRanker,
)
from backend.llm.utils import AnnotatedFloat


def _assert_approx(annotated_float: AnnotatedFloat, value: float | None, annotation: str) -> None:
    assert value == approx(annotated_float.value, abs=0.001)
    assert annotation == annotated_float.annotation


def test_naive_ranker() -> None:
    ranker = NaiveRanker(models=["a", "b"])
    assert ranker.rank_annotate("a") == AnnotatedFloat(None, "total_score=0.0, num_battles=0")
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(None, "[a vs b]: 0 battles; [b vs a]: 0 battles")

    ranker.update("a", "b", 1.0)
    assert ranker.rank_annotate("a") == AnnotatedFloat(1.0, "total_score=1.0, num_battles=1")
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(
        1.0, "[a vs b]: 1 battles; [b vs a]: 0 battles; sum(a)=1.00"
    )

    ranker.update("b", "a", 1.0)
    assert ranker.rank_annotate("a") == AnnotatedFloat(0.5, "total_score=1.0, num_battles=2")
    assert ranker.rank_annotate("b") == AnnotatedFloat(0.5, "total_score=1.0, num_battles=2")
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(
        0.5, "[a vs b]: 1 battles; [b vs a]: 1 battles; sum(a)=1.00"
    )

    ranker.update("a", "b", 1.0)
    ranker.update("b", "a", 0.0)
    assert ranker.rank_annotate("a") == AnnotatedFloat(0.75, "total_score=3.0, num_battles=4")
    assert ranker.rank_annotate("b") == AnnotatedFloat(0.25, "total_score=1.0, num_battles=4")
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(
        0.75, "[a vs b]: 2 battles; [b vs a]: 2 battles; sum(a)=3.00"
    )

    assert ranker.rank_annotate("c") == AnnotatedFloat(None, "total_score=0.0, num_battles=0")
    assert ranker.predict_annotate("a", "c") == AnnotatedFloat(None, "[a vs c]: 0 battles; [c vs a]: 0 battles")

    ranker.add_model("c")
    assert ranker.rank_annotate("c") == AnnotatedFloat(None, "total_score=0.0, num_battles=0")

    # Add an unrelated battle, ranking should change but prediction should not.
    ranker.update("a", "c", 1.0)
    assert ranker.rank_annotate("a") == AnnotatedFloat(0.80, "total_score=4.0, num_battles=5")
    assert ranker.rank_annotate("b") == AnnotatedFloat(0.25, "total_score=1.0, num_battles=4")
    assert ranker.rank_annotate("c") == AnnotatedFloat(0.00, "total_score=0.0, num_battles=1")
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(
        0.75, "[a vs b]: 2 battles; [b vs a]: 2 battles; sum(a)=3.00"
    )


def test_elo_ranker() -> None:
    ranker = EloRanker()

    assert ranker.rank_annotate("a") == AnnotatedFloat(
        1000, "Starting value 1000.0; 0 adjustments (0 negative, mean=0.00, stdev=0.00)"
    )
    assert ranker.predict_annotate("a", "b") == AnnotatedFloat(0.5, "rating_a=1000.00, rating_b=1000.00")

    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)
    ranker.update("a", "b", 1.0)

    _assert_approx(
        ranker.rank_annotate("a"), 1005.931, "Starting value 1000.0; 3 adjustments (0 negative, mean=1.98, stdev=0.02)"
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.517, "rating_a=1005.93, rating_b=994.07")

    ranker.update("a", "b", 0.0)
    ranker.update("a", "b", 0.0)

    _assert_approx(
        ranker.rank_annotate("a"), 1001.818, "Starting value 1000.0; 5 adjustments (2 negative, mean=0.36, stdev=1.98)"
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.505, "rating_a=1001.82, rating_b=998.18")

    for _ in range(1000):
        ranker.update("a", "b", 0.0)
    _assert_approx(
        ranker.rank_annotate("a"),
        672.676,
        "Starting value 1000.0; 1005 adjustments (1002 negative, mean=-0.33, stdev=0.39)",
    )
    _assert_approx(ranker.predict_annotate("a", "b"), 0.023, "rating_a=672.68, rating_b=1327.32")


@mark.filterwarnings("ignore:Mean of empty slice")
@mark.filterwarnings("ignore:invalid value encountered in scalar divide")
@mark.parametrize("algo", CHOIX_RANKER_ALGORITHMS)
def test_choix_ranker(algo: str) -> None:
    ranker = ChoixRanker(choix_ranker_algorithm=algo)

    assert ranker.rank_annotate("a") == AnnotatedFloat(None, "Wins: 0, Losses: 0, Ties: 0")
    with raises(ValueError):
        ranker.predict("a", "b")

    for _ in range(10):
        ranker.update("a", "b", 1.0)

    assert ranker.rank("a") > 3.5  # type: ignore
    assert ranker.rank("b") < -3.5  # type: ignore

    for _ in range(10):
        ranker.update("a", "b", 1.0)
    for _ in range(10):
        ranker.update("a", "b", 0.0)
    for _ in range(10):
        ranker.update("a", "b", 0.5)

    _assert_approx(ranker.rank_annotate("a"), 1081.093, "Wins: 20, Losses: 10, Ties: 10")
    _assert_approx(ranker.rank_annotate("b"), 918.906, "Wins: 10, Losses: 20, Ties: 10")
    _assert_approx(ranker.predict_annotate("a", "b"), 0.6, "rank_a=1081, rank_b=919")

    for _ in range(5):
        ranker.update("a", "b", 0.0)

    leaderboard = ranker.leaderboard()
    assert len(leaderboard) == 2
    assert leaderboard[0].model == "a"
    assert leaderboard[1].model == "b"
    assert leaderboard[0].rank.value == approx(1036.464, abs=0.01)
    assert leaderboard[1].rank.value == approx(963.535, abs=0.01)


def test_mab_ranker() -> None:
    ranker = MultiArmedBanditRanker(
        models=["a", "b"],
        learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.2),
    )
    ranker.fit([("a", "b"), ("a", "b"), ("a", "b"), ("a", "b")], [1.0, 0.0, 1.0, 1.0])

    assert ranker.rank("a") == 0.75
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
    rank, lower, upper = ci_ranker.rank_conf_intervals("a")
    assert lower == rank == upper == approx(2703.4786)  # no ambiguity

    for _ in range(5):
        ci_ranker.update("a", "b", 0.0)
        non_ci_ranker.update("a", "b", 0.0)
    rank, lower, upper = ci_ranker.rank_conf_intervals("a")
    assert lower == approx(1077.0497, abs=0.001)
    assert rank == approx(1202.256, abs=0.001)
    assert upper == approx(1374.191, abs=0.001)

    # ... and narrower as data is more consistent.
    for _ in range(50):
        ci_ranker.update("a", "b", 0.0)
        non_ci_ranker.update("a", "b", 0.0)
    rank, lower, upper = ci_ranker.rank_conf_intervals("a")
    assert lower == approx(602.914, abs=0.001)
    assert rank == approx(659.083, abs=0.001)
    assert upper == approx(713.847, abs=0.001)

    # Mean rank should be similar to the rank from a non-confidence-intervals ranker.
    assert ci_ranker.rank("a") == approx(non_ci_ranker.rank("a"), rel=0.03)
    assert ci_ranker.rank("b") == approx(non_ci_ranker.rank("b"), rel=0.03)


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
    expected_overall_ranks = {"a": approx(1081.026), "b": approx(918.973)}
    assert ranker.ranks() == expected_overall_ranks
    assert ranker.ranks_all_categories() == {
        # In coding, a is much better than b.
        "coding": {"a": approx(2381.751), "b": approx(-381.751)},
        # In math, b is much better than a.
        "math": {"b": approx(2381.751), "a": approx(-381.751)},
        "overall": expected_overall_ranks,
    }

    with raises(ValueError):
        ranker.update("a", "b", 1.0, category="non-existent category")

    rank_all_categories = ranker.rank_annotate_all_categories("a")

    assert len(rank_all_categories) == 3
    _assert_approx(rank_all_categories["coding"], 2381.751, "Wins: 2, Losses: 0, Ties: 0 (2381.8 to 2381.8)")
    _assert_approx(rank_all_categories["math"], -381.751, "Wins: 0, Losses: 2, Ties: 0 (-381.8 to -381.8)")
    _assert_approx(rank_all_categories["overall"], 1081.026, "Wins: 3, Losses: 2, Ties: 0 (811.2 to 2564.9)")

    assert ranker.annotate_prediction_all_categories("a", "b") == {
        "coding": "rank_a=2382, rank_b=-382",
        "math": "rank_a=-382, rank_b=2382",
        "overall": "rank_a=1081, rank_b=919",
    }
