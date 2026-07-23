"""
Atari environment preprocessing wrappers for the RL experiment.

Applies the standard preprocessing pipeline described in Appendix I.1:
  - Convert frames to grayscale
  - Resize to 84×84 pixels
  - Stack 4 consecutive frames
  - Clip rewards to [-1, 1]

Environment: BreakoutNoFrameskip-v4 / SeaquestNoFrameskip-v4
(RISK-07: version inferred from Huang et al. 2022 PPO codebase referenced in paper)

Paper: Appendix I.1. Mnih et al. (2015). Confidence: 0.92.
"""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformReward,
)


def make_atari_env(
    env_name: str,
    frame_size: int = 84,
    frame_stack: int = 4,
    reward_clip: float = 1.0,
    seed: int = 0,
    render_mode: str = None,
) -> gym.Env:
    """Create a preprocessed Atari environment.

    Applies the full preprocessing pipeline from Appendix I.1:
      1. Grayscale conversion
      2. Frame resize to frame_size × frame_size
      3. Frame stacking (4 frames)
      4. Reward clipping to [-reward_clip, reward_clip]

    Args:
        env_name:    Gymnasium environment ID (e.g. 'BreakoutNoFrameskip-v4').
        frame_size:  Target frame size (paper: 84). Default 84.
        frame_stack: Number of frames to stack (paper: 4). Default 4.
        reward_clip: Reward clipping magnitude (paper: 1.0). Default 1.0.
        seed:        Environment seed.
        render_mode: Optional render mode (e.g. 'human' for visualisation).

    Returns:
        Preprocessed gymnasium environment.
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=frame_size,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,    # Keep as uint8; divide by 255 in the model
    )
    env = FrameStackObservation(env, stack_size=frame_stack)
    env = TransformReward(env, lambda r: np.clip(r, -reward_clip, reward_clip))
    env.reset(seed=seed)
    return env


def make_vec_envs(
    env_name: str,
    num_envs: int,
    frame_size: int = 84,
    frame_stack: int = 4,
    reward_clip: float = 1.0,
    seed: int = 0,
) -> list:
    """Create a list of independent preprocessed Atari environments.

    Args:
        env_name:  Environment ID.
        num_envs:  Number of parallel environments.
        frame_size, frame_stack, reward_clip: Preprocessing parameters.
        seed:      Base seed; each env gets seed + i.

    Returns:
        List of preprocessed gym.Env instances.
    """
    return [
        make_atari_env(env_name, frame_size, frame_stack, reward_clip, seed=seed + i)
        for i in range(num_envs)
    ]
