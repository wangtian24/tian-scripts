import numpy as np


def get_battles(model_rewards: dict[str, float], num_samples: int) -> tuple[list[tuple[str, str]], list[float]]:
    """Get a list of battles and rewards from a set of models.

    Args:
        model_rewards: The rewards of the models, as a dictionary from model name to reward.
        num_samples: The number of battles to simulate.

    Returns:
        A list of battles and rewards. The battle models are selected propotionally to their rewards.
    """
    rng = np.random.default_rng(123)
    battles = []
    rewards = []
    probabilities = [reward / sum(model_rewards.values()) for reward in model_rewards.values()]
    for _ in range(num_samples):
        model_a, model_b = rng.choice(list(model_rewards.keys()), size=2, replace=False, p=probabilities)
        battles.append((str(model_a), str(model_b)))
        rewards.append(model_rewards[model_a])
    return battles, rewards
