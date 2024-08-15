from mabwiser.mab import LearningPolicy
from pytest import approx, mark, raises

from backend.llm.mab_ranker import MultiArmedBanditRanker
from backend.llm.ranking import (
    CHOIX_RANKER_ALGORITHMS,
    ChoixRanker,
    ChoixRankerConfIntervals,
    EloRanker,
    NaiveRanker,
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

    _assert_approx(ranker.rank_annotate("a"), 0.203, "Wins: 20, Losses: 10, Ties: 10")
    _assert_approx(ranker.rank_annotate("b"), -0.203, "Wins: 10, Losses: 20, Ties: 10")
    _assert_approx(ranker.predict_annotate("a", "b"), 0.6, "rank_a=0.20, rank_b=-0.20")

    for _ in range(5):
        ranker.update("a", "b", 0.0)

    leaderboard = ranker.leaderboard()
    assert len(leaderboard) == 2
    assert leaderboard[0].model == "a"
    assert leaderboard[1].model == "b"
    assert leaderboard[0].rank.value == approx(0.09, abs=0.01)
    assert leaderboard[1].rank.value == approx(-0.09, abs=0.01)


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
    ranker = ChoixRankerConfIntervals(
        models=["a", "b"],
        num_bootstrap_iterations=10,
        choix_ranker_algorithm="lsr_pairwise",
    )

    # The confidence intervals should get wider as data is more spread out.
    for _ in range(10):
        ranker.update("a", "b", 1.0)
    rank, lower, upper = ranker.rank_conf_intervals("a")
    assert lower == rank == upper == approx(4.2587)  # no ambiguity

    for _ in range(5):
        ranker.update("a", "b", 0.0)
    rank, lower, upper = ranker.rank_conf_intervals("a")
    assert lower == approx(0.1926, abs=0.001)
    assert rank == approx(0.5374, abs=0.001)
    assert upper == approx(0.9354, abs=0.001)

    # ... and narrower as data is more consistent.
    for _ in range(50):
        ranker.update("a", "b", 0.0)
    rank, lower, upper = ranker.rank_conf_intervals("a")
    assert lower == approx(-0.9927, abs=0.001)
    assert rank == approx(-0.8424, abs=0.001)
    assert upper == approx(-0.7153, abs=0.001)
