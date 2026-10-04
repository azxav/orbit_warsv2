from orbit_board_bc_train.local_match import classify_result, competition_rank, summarize_games


def test_competition_rank_ties_share_best_rank():
    assert competition_rank([3.0, 3.0, 1.0], 0) == 1
    assert competition_rank([3.0, 3.0, 1.0], 1) == 1
    assert competition_rank([3.0, 3.0, 1.0], 2) == 3


def test_classify_result_counts_shared_best_as_draw():
    assert classify_result([1.0, -1.0], 0) == "win"
    assert classify_result([1.0, -1.0], 1) == "loss"
    assert classify_result([1.0, 1.0], 0) == "draw"


def test_summarize_games_win_rate_and_average_rank():
    summary = summarize_games(
        [
            {"rewards": [1.0, -1.0], "seat": 0, "steps": 200},
            {"rewards": [1.0, -1.0], "seat": 1, "steps": 400},
            {"rewards": [1.0, 1.0], "seat": 0, "steps": 500},
        ]
    )
    assert summary["games"] == 3
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["win_rate"] == 1 / 3
    assert summary["average_rank"] == (1 + 2 + 1) / 3
    assert summary["average_reward"] == (1.0 + -1.0 + 1.0) / 3
    assert summary["average_steps"] == (200 + 400 + 500) / 3


def test_missing_reward_counts_as_a_loss():
    assert classify_result([None, 1.0], 0) == "loss"
    assert competition_rank([None, 1.0], 0) == 2
    assert classify_result([1.0, None], 0) == "win"
    summary = summarize_games([{"rewards": [None, 1.0], "seat": 0, "steps": 10}])
    assert summary["losses"] == 1
    assert summary["average_reward"] is None
    assert summary["average_rank"] == 2


def test_random_opponent_repeats_when_the_game_seed_is_fixed():
    pytest = __import__("pytest")
    pytest.importorskip("kaggle_environments")
    from orbit_board_bc_train.local_match import _play_game

    def noop(_obs):
        return []

    first = _play_game([noop, "random"], seed=7, episode_steps=12)
    second = _play_game([noop, "random"], seed=7, episode_steps=12)
    assert first[0] == second[0]
    assert first[2] == second[2]


def test_summarize_empty_games():
    summary = summarize_games([])
    assert summary["games"] == 0
    assert summary["win_rate"] is None
    assert summary["average_rank"] is None
