import wandb
import itertools
import json
import os
import torch
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from SAC import SACAgent
from DiscreteToContinuousWrapper import DiscreteToContinuousWrapper
from PPO import PPOAgent
from Actor_critic import A2CAgent
import numpy as np
import argparse
import random

# Ensure directories exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("best_configs", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Hyperparameter tuning strategy
TUNING_STRATEGY = "sequential"  # Options: "grid", "random", "sequential"
RANDOM_SAMPLES = 12  # Number of random samples if using random strategy
ENABLE_EARLY_STOPPING = True  # Prune bad runs early
EARLY_STOP_EPISODE = 75  # Check performance at this episode
EARLY_STOP_THRESHOLDS = {  # Minimum reward to continue
    "CartPole-v1": 80,  # Increased from 50 - should reach 100+ by episode 50 if learning
    "Acrobot-v1": -400,
    "MountainCar-v0": -199,
    "Pendulum-v1": -1400,
}

def get_hyperparameter_search_space(algo_name, env_name):
    """
    Returns algorithm- and environment-specific hyperparameter search ranges.
    Uses a Base + Override structure for optimal configurations.
    """
    # 1. Base Search Space (Safe defaults for standard problems)
    search_space = {
        'learning_rate': [1e-3, 5e-4, 3e-4],
        'gamma': [0.98, 0.99],
        'batch_size': [64],
        'buffer_size': [2048],  # memory_size in original spec
        'decay_rate': [0.99],
    }
    
    # 2. Algorithm Overrides (The Mechanism)
    if algo_name == "A2C":
        # On-Policy: Small Rollout Buffer, Smaller Batches
        search_space['buffer_size'] = [1024, 2048]
        search_space['batch_size'] = [32, 64]
        search_space['decay_rate'] = [0.99, 0.995]  # Entropy decay
    elif algo_name == "PPO":
        # On-Policy: Needs longer rollouts for stability
        search_space['buffer_size'] = [2048, 4096]
        search_space['batch_size'] = [64, 128]
        search_space['decay_rate'] = [0.9, 0.95, 0.99]  # Clip Range decay
    elif algo_name == "SAC":
        # Off-Policy: Huge Replay Buffer, Larger Batches
        search_space['buffer_size'] = [50000, 100000]
        search_space['batch_size'] = [128, 256]
        search_space['decay_rate'] = [1.0]  # SAC auto-tunes entropy
    
    # 3. Environment Overrides (The Problem Difficulty)
    if env_name == "CartPole-v1":
        # Short horizon, dense-ish rewards
        search_space['gamma'] = [0.98, 0.99]
        search_space['learning_rate'] = [1e-3, 5e-4]
    elif env_name == "Acrobot-v1":
        # Slightly harder, slower learning is safer
        search_space['gamma'] = [0.99]
        search_space['learning_rate'] = [5e-4, 3e-4]
    elif env_name == "MountainCar-v0":
        # The "Trap": Sparse rewards require extreme farsightedness and aggressive learning
        search_space['gamma'] = [0.995, 0.999]  # MUST be close to 1.0 for long-term credit
        search_space['learning_rate'] = [1e-3, 2e-3]  # Aggressive updates when signal arrives
        search_space['batch_size'] = [128, 256]  # Large batches to stabilize noisy gradients
        search_space['buffer_size'] = [2048] if algo_name != "SAC" else [50000, 100000]  # Capture full episodes
    elif env_name == "Pendulum-v1":
        # Continuous, smooth dynamics
        search_space['gamma'] = [0.99]
        search_space['learning_rate'] = [1e-3, 3e-4]
    
    return search_space

def get_sequential_configs(algo_name, env_name):
    """
    Sequential tuning: Tune one hyperparameter at a time, using the correct search space.
    Returns a smaller, focused set of configurations.
    """
    configs = []
    # Get the full, correct search space for the algo/env pair
    search_space = get_hyperparameter_search_space(algo_name, env_name)

    # Establish a base config using the FIRST value from each search space option
    base_config = {key: value[0] for key, value in search_space.items()}
    
    # Stage 1: Tune Learning Rate (most critical)
    for lr in search_space.get('learning_rate', [base_config['learning_rate']]):
        config = base_config.copy()
        config['learning_rate'] = lr
        configs.append(config)
    
    # Stage 2: Tune buffer/memory size with the established base LR
    for buf_size in search_space.get('buffer_size', [base_config['buffer_size']]):
        config = base_config.copy()
        config['buffer_size'] = buf_size
        configs.append(config)
    
    # Stage 3: Tune gamma with the established base LR
    for gamma in search_space.get('gamma', [base_config['gamma']]):
        config = base_config.copy()
        config['gamma'] = gamma
        configs.append(config)
        
    # Remove duplicates that might have been created
    unique_configs = [dict(t) for t in {tuple(d.items()) for d in configs}]
    
    return unique_configs

def make_env(env_name, model_type, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    # [cite: 11] SAC Wrapper Logic for Discrete Envs
    if model_type == "SAC" and env_name != "Pendulum-v1":
        env = DiscreteToContinuousWrapper(env)
    return env

def get_env_dims(env):
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    else:
        action_dim = env.action_space.n
        is_continuous = False
    return state_dim, action_dim, is_continuous

# --- PHASE 1: TRAINING & OPTIMIZATION ---
def train_and_validate(model_type, env_name, config):
    run_name = f"TRAIN_{model_type}_{env_name}_lr{config['learning_rate']}_bs{config['batch_size']}"
    wandb.init(project="CMPS458-Assignment3", config=config, reinit=True, name=run_name, group=f"{model_type}_{env_name}",mode="online")
    
    env = make_env(env_name, model_type)
    state_dim, action_dim, is_continuous = get_env_dims(env)
    
    # Instantiate Agent
    if model_type == "SAC":
        agent = SACAgent(state_dim, action_dim, env.action_space, config)
    elif model_type == "PPO":
        agent = PPOAgent(state_dim, action_dim, config, is_continuous)
    elif model_type == "A2C":
        agent = A2CAgent(state_dim, action_dim, config, is_continuous)

    # Train for 500 Episodes [cite: 12]
    print(f"   > Training {model_type} on {env_name}...")
    
    best_episode_reward = -float('inf')
    episode_rewards = []
    early_stopped = False

    for episode in range(500):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while not done:
            if model_type == "SAC":
                action = agent.select_action(state)
            else:
                action = agent.select_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            if model_type == "SAC":
                agent.store_transition(state, action, reward, next_state, done)
                if agent.memory.size > config['batch_size']:
                    loss = agent.update(config['batch_size'])
                    episode_losses.append(loss)
            elif model_type in ["PPO", "A2C"]:
                # For on-policy methods, use terminated (not truncated) for return calculation
                agent.store_reward(reward, terminated)
            
            state = next_state
            total_reward += reward

        # Decay entropy/alpha coefficient after each episode
        if model_type == "SAC" and hasattr(agent, "alpha"):
            if hasattr(agent, "alpha_decay"):
                agent.alpha *= config["decay_rate"]
        elif model_type in ["PPO", "A2C"] and hasattr(agent, "entropy_coef"):
            agent.entropy_coef *= config["decay_rate"]
            # Also update entropy_beta for A2C to keep them in sync
            if model_type == "A2C" and hasattr(agent, "entropy_beta"):
                agent.entropy_beta = agent.entropy_coef

        if model_type in ["PPO", "A2C"]:
            loss = agent.update()
            episode_losses.append(loss)
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # Track best episode reward
        if total_reward > best_episode_reward:
            best_episode_reward = total_reward
        
        # Early stopping check
        if ENABLE_EARLY_STOPPING and episode == EARLY_STOP_EPISODE:
            # MountainCar needs more time to learn - skip early stopping
            if env_name == "MountainCar-v0" and episode < 100:
                print(f"      ⏭ Skipping early stop for MountainCar (needs more exploration time)")
            else:
                threshold = EARLY_STOP_THRESHOLDS.get(env_name, -float('inf'))
                recent_avg = np.mean(episode_rewards[-min(20, len(episode_rewards)):])
                if recent_avg < threshold:
                    print(f"      ⚠ EARLY STOP at episode {episode}: Avg reward {recent_avg:.2f} < threshold {threshold}")
                    early_stopped = True
                    wandb.log({"early_stopped": 1, "stop_episode": episode, "stop_avg_reward": recent_avg})
                    break
                else:
                    print(f"      ✓ Passed early stop check: Avg reward {recent_avg:.2f} >= threshold {threshold}")
        
        # Log metrics to wandb
        log_dict = {
            "episode": episode,
            "train_reward": total_reward,
            "avg_reward_100": avg_reward,
            "best_reward": best_episode_reward,
            "episode_steps": steps,
        }
        
        if episode_losses:
            log_dict["train_loss"] = np.mean(episode_losses)
        
        if model_type in ["PPO", "A2C"]:
            log_dict["entropy_coef"] = agent.entropy_coef
        elif model_type == "SAC":
            log_dict["alpha"] = agent.alpha.item() if torch.is_tensor(agent.alpha) else agent.alpha
            log_dict["buffer_size"] = agent.memory.size
        
        wandb.log(log_dict)
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            print(f"      Episode {episode + 1}/500 | Reward: {total_reward:.2f} | Avg(100): {avg_reward:.2f} | Steps: {steps}")
    
    # If early stopped, return poor validation score
    if early_stopped:
        wandb.finish()
        return -float('inf'), agent

    # Short Validation to determine "Best" (5 Episodes)
    print(f"   > Validating {model_type} on {env_name}...")
    val_rewards = []
    for val_ep in range(5):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # Use evaluate=True for SAC to be deterministic
            if model_type == "SAC":
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        val_rewards.append(ep_reward)
        wandb.log({"val_reward": ep_reward, "val_episode": val_ep})
    
    avg_val_reward = np.mean(val_rewards)
    std_val_reward = np.std(val_rewards)
    
    print(f"   > Validation complete: Avg={avg_val_reward:.2f}, Std={std_val_reward:.2f}")
    wandb.log({"final_val_avg": avg_val_reward, "final_val_std": std_val_reward})
    wandb.finish()
    
    return avg_val_reward, agent

# --- PHASE 2: FINAL TESTING ---
def test_best_model(model_type, env_name, config, model_path):
    run_name = f"TEST_{model_type}_{env_name}_BEST"
    wandb.init(project="CMPS458-Assignment3", config=config, reinit=True, name=run_name, group="final_testing",mode="online")
    
    print(f"--- Starting Final Test: {model_type} on {env_name} ---")
    
    # 1. Setup Env & Agent
    # [cite: 18] RecordVideo Wrapper
    env = make_env(env_name, model_type, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=f"./videos/{model_type}_{env_name}", episode_trigger=lambda x: x % 19 == 0)
    
    state_dim, action_dim, is_continuous = get_env_dims(env)
    
    # 2. Re-instantiate Agent
    if model_type == "SAC":
        agent = SACAgent(state_dim, action_dim, env.action_space, config)
        checkpoint = torch.load(model_path)
        agent.policy.load_state_dict(checkpoint['policy'])
        # SAC critics aren't strictly needed for testing, but good practice to load if resumed
    elif model_type == "PPO":
        agent = PPOAgent(state_dim, action_dim, config, is_continuous)
        agent.policy.load_state_dict(torch.load(model_path))
    elif model_type == "A2C":
        agent = A2CAgent(state_dim, action_dim, config, is_continuous)
        agent.network.load_state_dict(torch.load(model_path))

    # 3. Run 100 Test Episodes [cite: 17]
    durations = []
    rewards = []
    
    print(f"   > Running 100 test episodes...")
    
    for i in range(100):
        state, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done:
            if model_type == "SAC":
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            ep_reward += reward
            
        durations.append(steps)
        rewards.append(ep_reward)
        wandb.log({
            "test_episode": i,
            "test_duration": steps,
            "test_reward": ep_reward
        })
        
        if (i + 1) % 20 == 0:
            print(f"      Test episode {i + 1}/100 | Reward: {ep_reward:.2f} | Steps: {steps}")
    
    avg_duration = np.mean(durations)
    std_duration = np.std(durations)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    print(f"--- Test Complete for {model_type} on {env_name} ---")
    print(f"    Avg Duration: {avg_duration:.2f} ± {std_duration:.2f}")
    print(f"    Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"    Max Reward: {max_reward:.2f} | Min Reward: {min_reward:.2f}")
    
    wandb.log({
        "final_avg_duration": avg_duration,
        "final_std_duration": std_duration,
        "final_avg_reward": avg_reward,
        "final_std_reward": std_reward,
        "final_max_reward": max_reward,
        "final_min_reward": min_reward
    })
    env.close()
    wandb.finish()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Assignment 3 Training Script")
    parser.add_argument("--algo", type=str, choices=["PPO", "A2C", "SAC"], help="Algorithm to use")
    parser.add_argument("--env", type=str, choices=["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"], help="Environment to use")
    args = parser.parse_args()

    # If arguments are provided, use them; else use all
    all_environments = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0", "Pendulum-v1"]
    all_models = ["PPO", "A2C", "SAC"]

    environments = [args.env] if args.env else all_environments
    models = [args.algo] if args.algo else all_models

    # Load existing registry if it exists, otherwise create new one
    registry_path = "best_configs/final_best_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            best_registry = json.load(f)
        print(f"📂 Loaded existing registry with {len(best_registry)} entries")
    else:
        best_registry = {}
        print("📝 Creating new registry")

    # ==========================
    # PHASE 1: OPTIMIZATION LOOP
    # ==========================
    print("\n=== PHASE 1: HYPERPARAMETER OPTIMIZATION ===")
    for env_name in environments:
        for model in models:
            best_score = -float('inf')

            print(f"\n> Optimizing {model} for {env_name}...")
            
            # Get algorithm- and environment-specific hyperparameter space
            if TUNING_STRATEGY == "sequential":
                configurations = get_sequential_configs(model, env_name)
                print(f"   Using SEQUENTIAL tuning: {len(configurations)} focused configurations")
            elif TUNING_STRATEGY == "random":
                hyperparams = get_hyperparameter_search_space(model, env_name)
                keys, values = zip(*hyperparams.items())
                all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
                configurations = random.sample(all_configs, min(RANDOM_SAMPLES, len(all_configs)))
                print(f"   Using RANDOM sampling: {len(configurations)} configurations")
            else:  # grid
                hyperparams = get_hyperparameter_search_space(model, env_name)
                keys, values = zip(*hyperparams.items())
                configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]
                print(f"   Using GRID search: {len(configurations)} configurations")

            for config in configurations:
                try:
                    print(f"   Testing config: lr={config['learning_rate']}, gamma={config['gamma']}, bs={config['batch_size']}, buffer={config['buffer_size']}")
                    score, agent = train_and_validate(model, env_name, config)
                    print(f"   ✓ Score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        
                        # Save Best Model
                        save_path = f"saved_models/{model}_{env_name}_best.pth"
                        agent.save(save_path)
                        
                        # Update Registry
                        best_registry[f"{model}_{env_name}"] = {
                            "config": config,
                            "path": save_path,
                            "score": best_score
                        }
                        print(f"   ★ NEW BEST! Score: {best_score:.2f} | Saved to {save_path}")
                        
                except Exception as e:
                    print(f"   ✗ Error with config {config}: {e}")
                    import traceback
                    traceback.print_exc()    # Save Registry to JSON for safety
    with open("best_configs/final_best_registry.json", "w") as f:
        json.dump(best_registry, f, indent=4)

    print("\n=== OPTIMIZATION COMPLETE. REGISTRY SAVED. ===")

    # ==========================
    # PHASE 2: FINAL TESTING
    # ==========================
    print("\n=== PHASE 2: FINAL TESTING & RECORDING ===")

    # Iterate through the registry of winners
    for key, data in best_registry.items():
        # Parse key "SAC_CartPole-v1" -> "SAC", "CartPole-v1"
        parts = key.split('_')
        model_type = parts[0]
        env_name = "_".join(parts[1:])

        config = data['config']
        path = data['path']

        try:
            test_best_model(model_type, env_name, config, path)
        except Exception as e:
            print(f"!!! Error testing {key}: {e}")

    print("\n=== ALL ASSIGNMENT TASKS COMPLETE ===")