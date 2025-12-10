# ASS4 - CarRacing and LunarLander encoder helpers

This folder provides a CNN encoder for `CarRacing-v3` and example actor/critic heads to integrate with your SAC/PPO/TD3 implementations.

Files:
- `encoders.py` - `CarRacingEncoder` (96x96 RGB -> feature vector)
- `actor_critic_carracing.py` - `Actor`, `QNetwork` using the encoder and simple MLP actor/critic for `LunarLander-v3` examples
- `example_usage.py` - quick shape tests to ensure everything imports and tensors flow correctly

Usage:
```bash
# Run the example checks
python ASS4/example_usage.py
```

Integration notes:
- Replace the initial fully-connected input layer in your existing Actor and Critic networks for `CarRacing-v3` with the `CarRacingEncoder` followed by an MLP head. The encoder outputs a `feature_dim` vector (default 256) that you can feed into your existing heads.
- For `LunarLander-v3`, you can keep your MLP architectures as-is (state dim = 8, action dim = 2 continuous).

Action ranges for CarRacing:
- Steering: -1 .. 1
- Gas: 0 .. 1
- Brake: 0 .. 1

The `Actor.sample()` method returns `tanh(z)`, producing values in (-1,1) for all components. Post-process gas and brake to map into [0,1] if needed (e.g., `(tanh_out[:,1:] + 1)/2`).

## Recommended PPO hyperparameters for LunarLander-v3 (continuous)

Use these settings as a copy-paste starting point. They follow SB3 / community best practices and work well with `VecNormalize(obs_only=True)` and multiple parallel environments (8 envs recommended):

```python
config = dict(
	learning_rate=3e-4,
	buffer_size=2048,    # SB3 `n_steps`
	batch_size=64,
	K_epochs=10,
	gamma=0.99,
	gae_lambda=0.95,
	eps_clip=0.2,
	entropy_coef=0.0,
	vf_coef=0.5,
	max_grad_norm=0.5,
	target_kl=0.03,
	policy_kwargs=dict(net_arch=[dict(pi=[256,256], vf=[256,256])])
)
```

Notes:
- Use `VecNormalize` for observation normalization (obs only). Do NOT stack frames.
- Run with 8 parallel environments for stable PPO gradients.
- Expected performance: average reward ~200+ when trained for a few hundred thousand timesteps.

## Run Commands

### Train LunarLander-v3 with PPO
```bash
# Basic training (500 episodes, then test on 100 episodes)
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500

# With Weights & Biases logging
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500 --use_wandb

# Skip testing after training
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500 --use_wandb --skip_test

# Custom number of test episodes
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500 --test_episodes 50 --use_wandb

# Record video after training
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500 --record_video --video_episodes 5
```

### Train CarRacing-v3 with PPO
```bash
# Basic training
python ASS4/train_ass4.py --algo PPO --env CarRacing-v3 --episodes 200

# With Weights & Biases logging and video recording
python ASS4/train_ass4.py --algo PPO --env CarRacing-v3 --episodes 200 --use_wandb --record_video
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--algo` | PPO | Algorithm to use (currently only PPO) |
| `--env` | CarRacing-v3 | Environment name |
| `--episodes` | 200 | Number of training episodes |
| `--test_episodes` | 100 | Number of test episodes after training |
| `--use_wandb` | False | Enable Weights & Biases logging |
| `--skip_test` | False | Skip testing after training |
| `--record_video` | False | Record video of trained agent |
| `--video_episodes` | 3 | Number of episodes to record |

### Output Folders
- `ASS4/saved_models/` - Saved model checkpoints (`.pth` files)
- `ASS4/best_configs/` - Best hyperparameter configs with test results (`.json` files)
- `ASS4/videos/` - Recorded videos of trained agents
python ASS4/train_ass4.py --algo PPO --env LunarLander-v3 --episodes 500 --use_wandb --record_video --video_episodes 5