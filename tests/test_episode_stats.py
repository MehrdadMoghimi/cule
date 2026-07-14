from cleanrl_utils.episode_stats import EpisodeStats


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, *args):
        self.scalars.append(args)


def test_episode_stats_solves_on_full_window():
    writer = _Writer()
    stats = EpisodeStats(window=2, solve_reward=1.5)

    assert not stats.update(
        {"final_info": [{"episode": {"r": 1.0, "l": 3}}, None]}, 10, writer
    )
    assert stats.update(
        {"final_info": [{"episode": {"r": 2.0, "l": 4}}]}, 20, writer
    )
    assert stats.mean_return == 1.5
    assert stats.solved_at_step == 20
    assert len(writer.scalars) == 6
