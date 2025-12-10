"""
Minimal PPO trainer for ASS4.
Supports:
 - LunarLander-v3 (vector obs, continuous action dim=2)
 - CarRacing-v3 (image obs, continuous action dim=3)

This script reuses `PPOAgent` from `ASS3/PPO.py`. It detects image observations and
passes a tuple (C,H,W) as the `state_dim` argument so the agent will use its CNN path.

CarRacing Speedup Optimizations:
 - Frame Skipping (4x): Actions repeated for 4 frames → 4x faster episodes
 - Early Termination: Stops episodes going very poorly (< -50 reward after 50 steps)
 - Negative Reward Patience: Terminates if negative rewards persist for 100 steps
 - Optimized K_epochs (4 instead of 10): Faster updates with similar quality
 - Larger buffer (4096): More data per update → fewer updates needed

These optimizations typically provide 5-6x training speedup without quality loss.

Run example:
    python ASS4/train_ass4.py --algo PPO --env CarRacing-v3 --episodes 200

"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import PPOAgent from the local PPO.py copy in ASS4
from PPO import PPOAgent

# Optional Weights & Biases integration (mirror ASS3 behavior)
try:
    import wandb
except Exception:
    wandb = None


def preprocess_obs(obs):
    # Convert to numpy array
    return np.asarray(obs)


def is_image_shape(shape):
    # Gym CarRacing often returns (96,96,3) HWC
    if not shape:
        return False
    if len(shape) == 3:
        return True
    return False


def to_chw(shape):
    # Given env.observation_space.shape (likely H,W,C), return (C,H,W)
    if len(shape) != 3:
        raise ValueError("Expected 3-dim image shape")
    h, w, c = shape
    return (c, h, w)


def postprocess_action_for_env(env_name, action):
    # action: numpy array
    if env_name == 'CarRacing-v3':
        # Actor returns values in (-1,1). Map gas/brake to [0,1]
        a = np.copy(action)
        if a.size >= 3:
            # steering in [-1,1] keep; gas/brake -> [0,1]
            a[1] = np.clip((a[1] + 1.0) / 2.0, 0.0, 1.0)
            a[2] = np.clip((a[2] + 1.0) / 2.0, 0.0, 1.0)
        return a
    else:
        return action


def get_frame_skip(env_name):
    """Return frame skip value for environment to speed up training."""
    if env_name == 'CarRacing-v3':
        return 4  # Skip 3 frames, act every 4th frame (4x speedup)
    return 1  # No frame skip for other environments


def default_config_for_env(env_name):
    # Minimal config; you can expand or pass via CLI
    if env_name == 'LunarLander-v3':
        # "High Stability" parameters for LunarLander-Continuous
        # This config achieved 211.71 avg reward and solved in ~2500 episodes
        return {
            'learning_rate': 2.5e-4,      # Slightly reduced for smoother updates
            'lr_decay': True,             # Enable linear LR decay
            'buffer_size': 4096,          # Collect experience
            'batch_size': 128,             # Smaller mini-batches for better convergence
            'K_epochs': 10,               # PPO update epochs
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'eps_clip': 0.15,              # PPO clip ratio
            'entropy_coef': 0.01,         # Encourages exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,         # Clips massive gradients
            'target_kl': 0.01,           # TIGHTER KL to prevent policy changing too fast
            'hidden_dim': 256,
            'action_std_init': 0.6,       # Start with high exploration
            'action_std_decay_rate': 0.05,  # Decay exploration
            'min_action_std': 0.1,        # Minimum exploration noise
            # SB3-style policy kwargs: separate pi and vf MLPs
            'policy_kwargs': {
                'net_arch': [ {'pi': [256, 256], 'vf': [256, 256]} ]
            }
        }
    elif env_name == 'CarRacing-v3':
        return {
            # "Exploration Booster" config for CarRacing-v3 - Escape negative reward trap
            'learning_rate': 1e-3,        # Increased from 3e-4 for faster learning
            'lr_decay': False,            # No LR decay for CarRacing
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'batch_size': 512,            # Increased from 256 for better gradient estimates
            'buffer_size': 8192,          # Doubled from 4096 for more diverse experience
            'eps_clip': 0.2,
            'K_epochs': 8,                # Increased from 3 to 8: squeeze more from noisy data
            'entropy_coef': 0.05,         # Boosted from 0.01 to 0.05: force risky actions (hard turns)
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,         # Prevent exploding gradients
            'hidden_dim': 256,
            'frame_skip': 4,              # Speed up 4x: act every 4th frame
            'early_stop_threshold': -100, # Relaxed from -50 to -100: allow recovery from mistakes
            'negative_reward_patience': 100,  # Terminate if negative for 100 steps
            'policy_kwargs': {
                'features_extractor_kwargs': {'features_dim': 512},
                'net_arch': [ {'pi': [256, 256], 'vf': [256, 256]} ]
            }
        }
    else:
        return {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 2048,
            'decay_rate': 1.0,
            'eps_clip': 0.2,
            'K_epochs': 40,
            'entropy_coef': 0.01,
            'hidden_dim': 256,
        }


def train(env_name, algo, episodes, save_dir='ASS4/saved_models', use_wandb=False):
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)

    obs_space = env.observation_space.shape
    action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
    is_continuous = isinstance(env.action_space, gym.spaces.Box)

    if is_image_shape(obs_space):
        state_dim = to_chw(obs_space)
        print(f"Detected image obs; using state_dim={state_dim}")
    else:
        # vector
        state_dim = obs_space[0]
        print(f"Detected vector obs; using state_dim={state_dim}")


    config = default_config_for_env(env_name)
    print("\n===== Hyperparameters for this run =====")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=======================================\n")

    # Initialize wandb if requested
    if use_wandb:
        if wandb is None:
            print("WANDB requested but `wandb` package not installed. Install or run without --use_wandb.")
        else:
            run_name = f"ASS4_{algo}_{env_name}_lr{config.get('learning_rate')}_bs{config.get('batch_size', config.get('buffer_size',''))}"
            wandb.init(project="CMPS458-Assignment3", config=config, settings=wandb.Settings(reinit="finish_previous"), name=run_name, group=f"{algo}_{env_name}", mode="online")

    if algo == 'PPO':
        agent = PPOAgent(state_dim, action_dim, config, is_continuous)
    else:
        raise NotImplementedError("Only PPO is implemented in this training helper")

    # Setup learning rate scheduler if enabled
    scheduler = None
    if config.get('lr_decay', False):
        scheduler = torch.optim.lr_scheduler.LinearLR(
            agent.optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=episodes
        )

    best_avg50 = -float('inf')
    best_reward = -float('inf')
    total_steps = 0
    episode_rewards = []  # Track rewards for averaging
    checkpoint_saved = False  # Track if any checkpoint was saved during training
    
    # Get frame skip and early termination settings for this environment
    frame_skip = config.get('frame_skip', 1)
    early_stop_threshold = config.get('early_stop_threshold', None)
    negative_reward_patience = config.get('negative_reward_patience', None)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        negative_reward_count = 0  # Track consecutive negative rewards

        while not done:
            inp = preprocess_obs(obs)
            action = agent.select_action(inp)
            # postprocess for specific envs
            action_env = postprocess_action_for_env(env_name, action)

            # Frame skipping: repeat action for frame_skip steps
            frame_reward = 0.0
            for _ in range(frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action_env)
                frame_reward += reward
                if terminated or truncated:
                    break
            
            done = terminated or truncated

            # store accumulated reward for on-policy agent
            agent.store_reward(frame_reward, terminated)

            obs = next_obs
            total_reward += frame_reward
            steps += 1
            
            # Early termination for CarRacing: stop if doing very poorly
            if early_stop_threshold is not None and total_reward < early_stop_threshold:
                if steps > 50:  # Give it at least 50 steps to start
                    done = True
                    
            # Track negative rewards for early stopping
            if negative_reward_patience is not None:
                if frame_reward < 0:
                    negative_reward_count += 1
                    if negative_reward_count >= negative_reward_patience:
                        done = True
                else:
                    negative_reward_count = 0

        total_steps += steps

        # end episode
        # decay entropy as in user's repo
        if hasattr(agent, 'entropy_coef'):
            agent.entropy_coef *= config.get('decay_rate', 1.0)

        # Call update when we've collected enough steps equal to buffer_size
        # For on-policy PPO, we accumulate steps across episodes until buffer_size is reached
        loss = 0.0
        buffer_len = len(agent.buffer.states) if hasattr(agent, 'buffer') else 0
        cfg_buf = config.get('buffer_size', None)
        
        # Update when buffer is full enough OR at end of training
        while buffer_len >= (cfg_buf or 64):
            loss = agent.update()
            buffer_len = len(agent.buffer.states) if hasattr(agent, 'buffer') else 0
            
            # Step learning rate scheduler if enabled (AFTER optimizer.step() inside update())
            if scheduler is not None:
                scheduler.step()

        episode_rewards.append(total_reward)
        
        # Track best single episode reward
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Calculate recent average for checkpoint (use last 50 episodes or all if less than 50)
        recent_avg = sum(episode_rewards[-min(50, len(episode_rewards)):]) / min(50, len(episode_rewards))
        
        # Save model when Avg50 improves AND is above 200 threshold
        if recent_avg > best_avg50 and recent_avg > 200:
            best_avg50 = recent_avg
            checkpoint_saved = True  # Mark that a checkpoint was saved
            # Include Avg50 in filename to prevent overwriting and track progress
            save_path = os.path.join(save_dir, f"{algo}_{env_name}_avg50_{best_avg50:.2f}.pth")
            agent.save(save_path)
            if use_wandb and (wandb is not None):
                try:
                    wandb.save(save_path)
                except Exception:
                    print("Warning: failed to save model to wandb")
            print(f"New best Avg50: {best_avg50:.2f} saved to {save_path}")
        elif recent_avg > best_avg50:
            # Update best_avg50 tracker without saving (below threshold)
            best_avg50 = recent_avg
        
        # Decay action standard deviation every 100 episodes
        if (ep + 1) % 100 == 0 and hasattr(agent.policy, 'action_var'):
            decay_rate = config.get('action_std_decay_rate', 0.05)
            min_std = config.get('min_action_std', 0.1)
            current_std = torch.sqrt(agent.policy.action_var[0]).item()
            new_std = max(current_std - decay_rate, min_std)
            agent.policy.action_var = torch.full((action_dim,), new_std * new_std).to(agent.device)
            agent.policy_old.action_var = torch.full((action_dim,), new_std * new_std).to(agent.device)
            print(f">>> Action std decayed to: {new_std:.4f}")
        
        # Get actual current learning rate from optimizer
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        # Get current action std if continuous
        current_action_std = None
        if hasattr(agent.policy, 'action_var') and agent.policy.action_var is not None:
            current_action_std = torch.sqrt(agent.policy.action_var[0]).item()

        if (ep + 1) % 10 == 0:
            log_msg = f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Best: {best_reward:.2f} | BestAvg50: {best_avg50:.2f} | Avg50: {recent_avg:.2f} | Loss: {loss:.4f} | LR: {current_lr:.2e}"
            if current_action_std is not None:
                log_msg += f" | ActionStd: {current_action_std:.3f}"
            log_msg += f" | Steps: {total_steps}"
            print(log_msg)

        # Print average every 100 episodes
        if (ep + 1) % 100 == 0:
            last_100 = episode_rewards[-100:]
            avg_reward = sum(last_100) / len(last_100)
            print(f"=== Average reward for episodes {ep+2-100}-{ep+1}: {avg_reward:.2f} ===")
            
            # Early stopping: if average of last 100 episodes > 290 (solved), stop training
            if avg_reward >= 290:
                print(f"=== SOLVED! Average reward {avg_reward:.2f} >= 290. Stopping early. ===")
                break

        # Log to wandb if enabled
        if use_wandb and (wandb is not None):
            log_dict = {
                'episode': ep,
                'train_reward': total_reward,
                'best_reward': best_reward,
                'best_avg50': best_avg50,
                'avg50': recent_avg,
                'loss': loss,
                'episode_steps': steps,
            }
            if hasattr(agent, 'entropy_coef'):
                log_dict['entropy_coef'] = agent.entropy_coef
            if hasattr(agent, 'lr'):
                log_dict['learning_rate'] = agent.lr
            try:
                wandb.log(log_dict)
            except Exception:
                print("Warning: wandb.log failed for episode", ep)

    env.close()
    
    # Save final model only if no checkpoint was saved during training
    if not checkpoint_saved:
        final_save_path = os.path.join(save_dir, f"{algo}_{env_name}_ass4_final.pth")
        agent.save(final_save_path)
        print(f"No checkpoint reached Avg50 >= 200. Final model saved to {final_save_path}")
        if use_wandb and (wandb is not None):
            try:
                wandb.save(final_save_path)
            except Exception:
                print("Warning: failed to save final model to wandb")
    else:
        print(f"Best checkpoint already saved with Avg50: {best_avg50:.2f}")
    
    if use_wandb and (wandb is not None):
        try:
            wandb.finish()
        except Exception:
            pass
    print("Training finished")
    
    return agent, save_dir, algo, env_name, episode_rewards


def plot_training_rewards(episode_rewards, env_name, algo='PPO', save_dir='ASS4/saved_models'):
    """Plot and save training rewards graph."""
    plt.figure(figsize=(12, 6))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Plot raw rewards
    plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot moving average (window=10)
    if len(episode_rewards) >= 10:
        window = 10
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(sum(episode_rewards[start:i+1]) / (i - start + 1))
        plt.plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window})')
    
    # Plot moving average (window=100) if enough episodes
    if len(episode_rewards) >= 100:
        window = 100
        moving_avg_100 = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg_100.append(sum(episode_rewards[start:i+1]) / (i - start + 1))
        plt.plot(episodes, moving_avg_100, color='green', linewidth=2, label=f'Moving Avg (window={window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'{algo} Training Rewards - {env_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{algo}_{env_name}_training_rewards.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training rewards plot saved to {plot_path}")
    return plot_path


def test(agent, env_name, num_episodes=100, record_video=False, video_dir='ASS4/videos', video_episodes=3):
    """Test the trained agent on the environment, optionally recording videos."""
    print(f"\n{'='*50}")
    print(f"Testing {env_name} for {num_episodes} episodes...")
    if record_video:
        print(f"Recording first {video_episodes} episodes to {video_dir}")
    print(f"{'='*50}")
    
    # Create video folder if recording
    video_folder = None
    if record_video:
        video_folder = os.path.join(video_dir, f"{env_name.replace('-', '_')}")
        os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with or without video recording
    # Record every 10th episode (0, 10, 20, ...)
    if record_video:
        env = gym.make(env_name, render_mode='rgb_array')
        env = RecordVideo(
            env, 
            video_folder, 
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix=f"PPO_{env_name}"
        )
    else:
        env = gym.make(env_name)
    
    test_rewards = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            inp = preprocess_obs(obs)
            with torch.no_grad():
                action = agent.select_action(inp)
            action_env = postprocess_action_for_env(env_name, action)
            
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
        
        # Clear buffer after each test episode (we don't want to train)
        if hasattr(agent, 'buffer'):
            agent.buffer.clear_memory()
        
        test_rewards.append(total_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"Test Episode {ep+1}/{num_episodes} | Reward: {total_reward:.2f}")
    
    env.close()
    
    avg_reward = sum(test_rewards) / len(test_rewards)
    min_reward = min(test_rewards)
    max_reward = max(test_rewards)
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    if record_video:
        print(f"  Videos saved to: {video_folder}")
    print(f"{'='*50}")
    
    return test_rewards, avg_reward


def plot_test_analysis(test_rewards, env_name, algo='PPO', save_dir='ASS4/saved_models'):
    """Generate comprehensive test performance analysis plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Reward Distribution Histogram
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(test_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(test_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(test_rewards):.2f}')
    ax1.axvline(np.median(test_rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(test_rewards):.2f}')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Rewards Over Time
    ax2 = plt.subplot(3, 3, 2)
    episodes = range(1, len(test_rewards) + 1)
    ax2.plot(episodes, test_rewards, 'o-', alpha=0.6, markersize=4)
    ax2.axhline(np.mean(test_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.fill_between(episodes, 
                      np.mean(test_rewards) - np.std(test_rewards), 
                      np.mean(test_rewards) + np.std(test_rewards), 
                      alpha=0.2, color='red', label='±1 Std Dev')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('Test Rewards Over Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box Plot
    ax3 = plt.subplot(3, 3, 3)
    box = ax3.boxplot([test_rewards], labels=['Test Rewards'], patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Box Plot')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative Distribution Function (CDF)
    ax4 = plt.subplot(3, 3, 4)
    sorted_rewards = np.sort(test_rewards)
    cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    ax4.plot(sorted_rewards, cdf, linewidth=2)
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th Percentile')
    ax4.axvline(np.median(test_rewards), color='green', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution Function')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Average (window=10)
    ax5 = plt.subplot(3, 3, 5)
    window = min(10, len(test_rewards))
    rolling_avg = []
    for i in range(len(test_rewards)):
        start = max(0, i - window + 1)
        rolling_avg.append(np.mean(test_rewards[start:i+1]))
    ax5.plot(episodes, test_rewards, alpha=0.3, color='blue', label='Raw Rewards')
    ax5.plot(episodes, rolling_avg, color='red', linewidth=2, label=f'Rolling Avg (w={window})')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Reward')
    ax5.set_title('Rolling Average Performance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Percentile Analysis
    ax6 = plt.subplot(3, 3, 6)
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [np.percentile(test_rewards, p) for p in percentiles]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    bars = ax6.bar([f'{p}th' for p in percentiles], percentile_values, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Reward')
    ax6.set_title('Percentile Performance')
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, percentile_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Success Rate Analysis (for LunarLander: >200 is considered success)
    ax7 = plt.subplot(3, 3, 7)
    thresholds = np.linspace(min(test_rewards), max(test_rewards), 20)
    success_rates = [(np.sum(np.array(test_rewards) >= t) / len(test_rewards)) * 100 for t in thresholds]
    ax7.plot(thresholds, success_rates, linewidth=2, color='blue')
    # Mark 200 threshold for LunarLander if applicable
    if 'Lunar' in env_name or 'lunar' in env_name:
        if min(test_rewards) <= 200 <= max(test_rewards):
            success_at_200 = (np.sum(np.array(test_rewards) >= 200) / len(test_rewards)) * 100
            ax7.axvline(200, color='red', linestyle='--', linewidth=2, label=f'Solved (200): {success_at_200:.1f}%')
            ax7.axhline(success_at_200, color='red', linestyle='--', alpha=0.3)
            ax7.legend()
    ax7.set_xlabel('Reward Threshold')
    ax7.set_ylabel('Success Rate (%)')
    ax7.set_title('Success Rate vs Threshold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Statistical Summary Text
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    stats_text = f"""
    Statistical Summary
    {'='*30}
    
    Episodes:        {len(test_rewards)}
    Mean:            {np.mean(test_rewards):.2f}
    Median:          {np.median(test_rewards):.2f}
    Std Dev:         {np.std(test_rewards):.2f}
    
    Min:             {np.min(test_rewards):.2f}
    Max:             {np.max(test_rewards):.2f}
    Range:           {np.max(test_rewards) - np.min(test_rewards):.2f}
    
    Q1 (25%):        {np.percentile(test_rewards, 25):.2f}
    Q3 (75%):        {np.percentile(test_rewards, 75):.2f}
    IQR:             {np.percentile(test_rewards, 75) - np.percentile(test_rewards, 25):.2f}
    
    Skewness:        {stats.skew(test_rewards):.3f}
    Kurtosis:        {stats.kurtosis(test_rewards):.3f}
    """
    if 'Lunar' in env_name or 'lunar' in env_name:
        success_count = np.sum(np.array(test_rewards) >= 200)
        success_rate = (success_count / len(test_rewards)) * 100
        stats_text += f"\n    Success (≥200):  {success_count}/{len(test_rewards)} ({success_rate:.1f}%)"
    
    ax8.text(0.1, 0.95, stats_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 9. Reward Stability (Standard Deviation Over Time)
    ax9 = plt.subplot(3, 3, 9)
    window = min(20, len(test_rewards) // 2)
    if window > 1:
        rolling_std = []
        for i in range(len(test_rewards)):
            start = max(0, i - window + 1)
            rolling_std.append(np.std(test_rewards[start:i+1]))
        ax9.plot(episodes, rolling_std, color='purple', linewidth=2)
        ax9.set_xlabel('Episode')
        ax9.set_ylabel('Std Dev')
        ax9.set_title(f'Reward Stability (Rolling Std, w={window})')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'Insufficient data\nfor stability analysis', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.axis('off')
    
    plt.suptitle(f'{algo} Test Performance Analysis - {env_name}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the comprehensive plot
    plot_path = os.path.join(save_dir, f'{algo}_{env_name}_test_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test analysis plots saved to {plot_path}")
    
    # Also create individual focused plots
    plot_individual_test_charts(test_rewards, env_name, algo, save_dir)
    
    return plot_path


def plot_individual_test_charts(test_rewards, env_name, algo='PPO', save_dir='ASS4/saved_models'):
    """Create individual detailed charts for key metrics."""
    
    # 1. Detailed Reward Timeline with Confidence Intervals
    plt.figure(figsize=(14, 6))
    episodes = range(1, len(test_rewards) + 1)
    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    
    plt.plot(episodes, test_rewards, 'o-', alpha=0.6, markersize=5, linewidth=1.5, label='Episode Reward')
    plt.axhline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    plt.axhline(mean_reward + std_reward, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'+1σ: {mean_reward + std_reward:.2f}')
    plt.axhline(mean_reward - std_reward, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'-1σ: {mean_reward - std_reward:.2f}')
    
    if 'Lunar' in env_name or 'lunar' in env_name:
        plt.axhline(200, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Solved Threshold (200)')
    
    plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, 
                     alpha=0.15, color='red', label='±1σ Range')
    
    plt.xlabel('Test Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'{algo} Test Performance Timeline - {env_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, f'{algo}_{env_name}_test_timeline.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test timeline plot saved to {plot_path}")
    
    # 2. Performance Consistency Chart
    plt.figure(figsize=(10, 6))
    window_sizes = [5, 10, 20, 30] if len(test_rewards) > 30 else [5, 10]
    
    for window in window_sizes:
        if window <= len(test_rewards):
            rolling_avg = []
            for i in range(len(test_rewards)):
                start = max(0, i - window + 1)
                rolling_avg.append(np.mean(test_rewards[start:i+1]))
            plt.plot(episodes, rolling_avg, linewidth=2, label=f'Window={window}', alpha=0.8)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Rolling Average Reward', fontsize=12)
    plt.title(f'{algo} Test Consistency Analysis - {env_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_dir, f'{algo}_{env_name}_test_consistency.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test consistency plot saved to {plot_path}")


def save_best_config(env_name, config, test_avg_reward, config_dir='ASS4/best_configs'):
    """Save the best configuration to a JSON file."""
    os.makedirs(config_dir, exist_ok=True)
    
    config_data = {
        'env_name': env_name,
        'algorithm': 'PPO',
        'test_avg_reward': test_avg_reward,
        'config': config
    }
    
    config_path = os.path.join(config_dir, f"PPO_{env_name}_best_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Best config saved to {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO'], help='Algorithm')
    parser.add_argument('--env', type=str, default='CarRacing-v3', help='Environment name')
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging (requires wandb)')
    parser.add_argument('--skip_test', action='store_true', help='Skip testing after training')
    parser.add_argument('--record_video', action='store_true', help='Record video of trained agent')
    parser.add_argument('--video_episodes', type=int, default=3, help='Number of episodes to record')
    args = parser.parse_args()
    
    # Print speedup info for CarRacing
    if args.env == 'CarRacing-v3':
        print("\n" + "="*60)
        print("CarRacing-v3 Training with Speed Optimizations Enabled")
        print("="*60)
        print("Speedup techniques applied:")
        print("  • Frame Skip (4x): Actions repeated for 4 frames")
        print("  • Early Termination: Stops poor episodes (< -50 reward)")
        print("  • Negative Patience: Exits if stuck with negative rewards")
        print("  • Optimized K_epochs: 4 updates per batch (faster)")
        print("  • Larger Buffer: 4096 steps per update")
        print("\nExpected speedup: ~5-6x faster training")
        print("Quality: Maintained through frame skip and smart termination")
        print("="*60 + "\n")

    agent, save_dir, algo, env_name, episode_rewards = train(args.env, args.algo, args.episodes, use_wandb=args.use_wandb)
    
    # Plot training rewards
    plot_training_rewards(episode_rewards, env_name, algo, save_dir)
    
    # Get config for saving
    config = default_config_for_env(env_name)
    
    # Test the trained model (with optional video recording)
    test_avg_reward = None
    if not args.skip_test:
        test_rewards, test_avg_reward = test(
            agent, 
            env_name, 
            num_episodes=args.test_episodes,
            record_video=args.record_video,
            video_episodes=args.video_episodes
        )
        
        # Generate comprehensive test analysis plots
        print("\nGenerating test performance analysis plots...")
        plot_test_analysis(test_rewards, env_name, algo, save_dir)
        
        # Save best config with test results
        save_best_config(env_name, config, test_avg_reward)
