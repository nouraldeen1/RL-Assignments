import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Normal
import numpy as np
import numpy as np

class PPOMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous, hidden_dim=256, action_std_init=0.6, encoder_feature_dim=256, net_arch=None):
        super(ActorCritic, self).__init__()
        self.is_continuous = is_continuous
        self.action_dim = action_dim
        self.is_image = isinstance(state_dim, (tuple, list))

        if self.is_image:
            # state_dim can be (H,W,C) or (C,H,W). Detect channels.
            shape = tuple(state_dim)
            if len(shape) != 3:
                raise ValueError("Image observations must have 3 dims (H,W,C) or (C,H,W)")
            # If first dim is channels (1 or 3), use it, otherwise assume last dim is channels
            if shape[0] in (1, 3):
                in_channels = shape[0]
            else:
                in_channels = shape[-1]

            # OPTIMIZED: Smaller CNN with fewer parameters for faster training
            # Reduced channels: 32→16, 64→32, 128→64, 256→128 (4x fewer params)
            # Still maintains good feature extraction capacity
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Project conv features to encoder_feature_dim (like features_extractor.features_dim)
            # Compute conv_out_size dynamically by passing a dummy tensor through the encoder.
            # This ensures the linear layer matches the actual flattened conv output
            # even when input image size changes (e.g. downsampled to 64x64).
            try:
                # Determine input spatial dims from state_dim tuple
                if shape[0] in (1, 3):
                    _, H, W = shape
                else:
                    H, W, _ = shape
                dummy = torch.zeros(1, in_channels, H, W)
                with torch.no_grad():
                    conv_out = self.encoder(dummy)
                conv_out_size = int(conv_out.view(1, -1).size(1))
            except Exception:
                # Fallback to conservative default if something goes wrong
                conv_out_size = 128 * 4 * 4

            self.encoder_proj = nn.Sequential(
                nn.Linear(conv_out_size, encoder_feature_dim),
                nn.Tanh(),
            )

            # Use net_arch if provided (SB3 style: dict with keys 'pi' and 'vf')
            pi_arch = None
            vf_arch = None
            if isinstance(net_arch, dict):
                pi_arch = net_arch.get('pi', None)
                vf_arch = net_arch.get('vf', None)

            feat_dim = encoder_feature_dim
            # Build actor head
            actor_layers = []
            last_dim = feat_dim
            arch = pi_arch if pi_arch is not None else [hidden_dim, hidden_dim]
            for h in arch:
                actor_layers.append(nn.Linear(last_dim, h))
                actor_layers.append(nn.Tanh())
                last_dim = h
            actor_layers.append(nn.Linear(last_dim, action_dim))
            # For continuous actions we output the mean (unbounded) and perform
            # tanh squashing after sampling; do NOT append final Tanh here.
            if is_continuous:
                self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
            else:
                actor_layers.append(nn.Softmax(dim=-1))
            self.actor = nn.Sequential(*actor_layers)

            # Build critic head
            critic_layers = []
            last_dim = feat_dim
            arch = vf_arch if vf_arch is not None else [hidden_dim, hidden_dim]
            for h in arch:
                critic_layers.append(nn.Linear(last_dim, h))
                critic_layers.append(nn.Tanh())
                last_dim = h
            critic_layers.append(nn.Linear(last_dim, 1))
            self.critic = nn.Sequential(*critic_layers)

        else:
            # MLP path for vector observations - use net_arch if provided
            pi_arch = None
            vf_arch = None
            if isinstance(net_arch, dict):
                pi_arch = net_arch.get('pi', None)
                vf_arch = net_arch.get('vf', None)

            if is_continuous:
                self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
                # Build actor with configurable architecture
                actor_layers = []
                last_dim = state_dim
                arch = pi_arch if pi_arch is not None else [hidden_dim, hidden_dim]
                for h in arch:
                    actor_layers.append(nn.Linear(last_dim, h))
                    actor_layers.append(nn.Tanh())
                    last_dim = h
                actor_layers.append(nn.Linear(last_dim, action_dim))
                self.actor = nn.Sequential(*actor_layers)
            else:
                actor_layers = []
                last_dim = state_dim
                arch = pi_arch if pi_arch is not None else [hidden_dim, hidden_dim]
                for h in arch:
                    actor_layers.append(nn.Linear(last_dim, h))
                    actor_layers.append(nn.Tanh())
                    last_dim = h
                actor_layers.append(nn.Linear(last_dim, action_dim))
                actor_layers.append(nn.Softmax(dim=-1))
                self.actor = nn.Sequential(*actor_layers)

            # Critic with configurable architecture
            critic_layers = []
            last_dim = state_dim
            arch = vf_arch if vf_arch is not None else [hidden_dim, hidden_dim]
            for h in arch:
                critic_layers.append(nn.Linear(last_dim, h))
                critic_layers.append(nn.Tanh())
                last_dim = h
            critic_layers.append(nn.Linear(last_dim, 1))
            self.critic = nn.Sequential(*critic_layers)
        
        # OPTIMIZATION: Orthogonal initialization for better training stability
        # This helps with gradient flow and faster convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply orthogonal initialization to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal initialization for hidden layers
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                # Orthogonal initialization for conv layers
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def act(self, state, device):
        # Handle image inputs (ensure shape is BxCxHxW)
        if self.is_image:
            # if incoming state is a single image tensor without batch dim
            if state.dim() == 3:
                # try to detect channel position
                if state.shape[0] in (1, 3):
                    img = state.unsqueeze(0)
                else:
                    # assume HWC -> CHW
                    img = state.permute(2, 0, 1).unsqueeze(0)
            else:
                img = state

            # CRITICAL: Normalize image to [0, 1] range
            if img.max() > 1.0:
                img = img / 255.0
            
            feats_conv = self.encoder(img.to(device))
            feats = self.encoder_proj(feats_conv)
            if self.is_continuous:
                # Squashed Gaussian: sample pre-tanh, then tanh squashing
                action_mean = self.actor(feats)
                action_std = torch.sqrt(self.action_var).to(device)
                normal = Normal(action_mean, action_std)
                pre_tanh = normal.rsample()
                action = torch.tanh(pre_tanh)
                # log_prob with change of variables: log N(pre_tanh) - sum log(1 - tanh(pre_tanh)^2)
                log_prob_pre = normal.log_prob(pre_tanh).sum(-1)
                # numerical stability
                eps = 1e-6
                log_det = torch.log(1 - action.pow(2) + eps).sum(-1)
                action_logprob = log_prob_pre - log_det
                state_val = self.critic(feats)
            else:
                action_probs = self.actor(feats)
                dist = Categorical(action_probs)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                state_val = self.critic(feats)

            if self.is_continuous:
                return action.detach().squeeze(0), action_logprob.detach().squeeze(0), state_val.detach().squeeze(0), pre_tanh.detach().squeeze(0)
            else:
                return action.detach().squeeze(0), action_logprob.detach().squeeze(0), state_val.detach().squeeze(0)

        else:
            if self.is_continuous:
                action_mean = self.actor(state)
                action_std = torch.sqrt(self.action_var).to(device)
                normal = Normal(action_mean, action_std)
                pre_tanh = normal.rsample()
                action = torch.tanh(pre_tanh)
                log_prob_pre = normal.log_prob(pre_tanh).sum(-1)
                eps = 1e-6
                log_det = torch.log(1 - action.pow(2) + eps).sum(-1)
                action_logprob = log_prob_pre - log_det
                state_val = self.critic(state)
                return action.detach(), action_logprob.detach(), state_val.detach(), pre_tanh.detach()
            else:
                action_probs = self.actor(state)
                dist = Categorical(action_probs)

            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)

            return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action, device):
        # Evaluate supports both image and vector states
        if self.is_image:
            # state shape expected: (N, C, H, W) or (N, H, W, C)
            if state.dim() == 4 and state.shape[1] not in (1, 3):
                # assume NHWC -> NCHW
                state = state.permute(0, 3, 1, 2)
            
            # CRITICAL: Normalize image to [0, 1] range
            if state.max() > 1.0:
                state = state / 255.0

            feats_conv = self.encoder(state.to(device))
            feats = self.encoder_proj(feats_conv)
            if self.is_continuous:
                action_mean = self.actor(feats)
                action_var = self.action_var.expand_as(action_mean).to(device)
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
                if self.action_dim == 1:
                    action = action.reshape(-1, self.action_dim)
            else:
                action_probs = self.actor(feats)
                dist = Categorical(action_probs)

            if self.is_continuous:
                # create Normal distribution corresponding to the predicted mean and std
                action_mean = self.actor(feats)
                action_std = torch.sqrt(self.action_var).expand_as(action_mean).to(device)
                normal = Normal(action_mean, action_std)
                # action is expected to be pre_tanh here (stored that way)
                log_prob_pre = normal.log_prob(action).sum(-1)
                eps = 1e-6
                action_tanh = torch.tanh(action)
                log_det = torch.log(1 - action_tanh.pow(2) + eps).sum(-1)
                action_logprobs = log_prob_pre - log_det
                # approximate entropy: use Gaussian entropy (before tanh)
                dist_entropy = normal.entropy().sum(-1)
                state_values = self.critic(feats)
                return action_logprobs, state_values, dist_entropy
            else:
                action_logprobs = dist.log_prob(action)
                dist_entropy = dist.entropy()
                # Use extracted features for value prediction, not the raw image tensor
                state_values = self.critic(feats)
                return action_logprobs, state_values, dist_entropy

        else:
            if self.is_continuous:
                action_mean = self.actor(state)
                action_std = torch.sqrt(self.action_var).expand_as(action_mean).to(device)
                normal = Normal(action_mean, action_std)
                if self.action_dim == 1:
                    action = action.reshape(-1, self.action_dim)
                # action is pre_tanh
                log_prob_pre = normal.log_prob(action).sum(-1)
                eps = 1e-6
                action_tanh = torch.tanh(action)
                log_det = torch.log(1 - action_tanh.pow(2) + eps).sum(-1)
                action_logprobs = log_prob_pre - log_det
                dist_entropy = normal.entropy().sum(-1)
                state_values = self.critic(state)
                return action_logprobs, state_values, dist_entropy
            else:
                action_probs = self.actor(state)
                dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(state)

            return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, config, is_continuous):
        self.lr = config.get('learning_rate', 1e-3)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.K_epochs = config.get('K_epochs', 40)
        self.entropy_coef = config.get('entropy_coef', 0.01)  # Entropy coefficient for exploration
        # Additional PPO hyperparameters (may be provided by config)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.gae_lambda = config.get('gae_lambda', 1.0)
        self.batch_size = config.get('batch_size', None)
        self.buffer_size = config.get('buffer_size', None)
        self.max_grad_norm = config.get('max_grad_norm', None)
        self.target_kl = config.get('target_kl', None)
        hidden_dim = config.get('hidden_dim', 256)
        action_std_init = config.get('action_std_init', 0.6)
        # policy_kwargs support (SB3-style)
        policy_kwargs = config.get('policy_kwargs', {}) or {}
        features_extractor_kwargs = policy_kwargs.get('features_extractor_kwargs', {}) or {}
        encoder_feature_dim = features_extractor_kwargs.get('features_dim', 256)
        net_arch = policy_kwargs.get('net_arch', None)
        # SB3 sometimes provides net_arch as a list like [dict(pi=[..], vf=[..])]
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            net_arch = net_arch[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> Using device: {self.device}")
        if torch.cuda.is_available():
            print(f">>> GPU: {torch.cuda.get_device_name(0)}")
        
        self.buffer = PPOMemory()
        # Pass encoder_feature_dim and net_arch into ActorCritic when applicable
        self.policy = ActorCritic(state_dim, action_dim, is_continuous, hidden_dim, action_std_init, encoder_feature_dim, net_arch).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_dim, action_dim, is_continuous, hidden_dim, action_std_init, encoder_feature_dim, net_arch).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.SmoothL1Loss()
    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
    
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if self.policy.is_continuous:
                action, action_logprob, _, pre_tanh = self.policy_old.act(state, self.device)
            else:
                action, action_logprob, _ = self.policy_old.act(state, self.device)

        self.buffer.states.append(state)
        # For continuous actions we store the pre-tanh value so evaluate() can compute
        # correct log-probs later. For discrete we store the action as-is.
        if self.policy.is_continuous:
            self.buffer.actions.append(pre_tanh)
        else:
            self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        if self.policy.is_continuous:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()
            
    # Added helper to match generic calls (optional but good for consistency)
    def store_reward(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def update(self):
        # Check if buffer has data
        if len(self.buffer.states) == 0:
            return 0.0

        # Convert buffer to tensors
        # CRITICAL: Don't use squeeze() for images - it removes channel dimension!
        # For images with shape (N, 1, H, W) squeeze would give (N, H, W) - WRONG!
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        if old_states.dim() == 2:  # Only squeeze for vector observations
            old_states = torch.squeeze(old_states)
        
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        
        # OPTIMIZATION: Reward normalization for stable learning
        # Normalize rewards to reduce variance (only if we have enough samples)
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Get state values for all states (no grad)
        with torch.no_grad():
            _, state_values, _ = self.policy.evaluate(old_states, old_actions, self.device)
        state_values = state_values.view(-1).detach()

        # Compute GAE advantages
        advantages = torch.zeros_like(rewards).to(self.device)
        last_adv = 0.0
        # iterate reversed
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - is_terminals[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - is_terminals[t]
                next_value = state_values[t+1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - state_values[t]
            last_adv = delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv

        returns = advantages + state_values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # If batch_size not provided, use full-batch (as previous behavior)
        batch_size = self.batch_size if (self.batch_size is not None and self.batch_size > 0) else len(rewards)
        total_steps = len(rewards)

        # Optimize policy for K epochs using minibatches
        loss_item = 0.0
        for epoch in range(self.K_epochs):
            # create random permutation of indices
            idxs = torch.randperm(total_steps)
            for start in range(0, total_steps, batch_size):
                mb_idx = idxs[start:start+batch_size]
                mb_states = old_states[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logprobs_new, state_values_new, dist_entropy = self.policy.evaluate(mb_states, mb_actions, self.device)
                # Ensure state_values_new has same shape as mb_returns
                state_values_new = state_values_new.view(-1)
                mb_returns_flat = mb_returns.view(-1)

                ratios = torch.exp(logprobs_new - mb_logprobs.detach())
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # OPTIMIZATION: Value function clipping for stability (like PPO paper)
                # Prevents value function from changing too drastically
                mb_old_values = state_values[mb_idx]
                value_pred_clipped = mb_old_values + torch.clamp(
                    state_values_new - mb_old_values,
                    -self.eps_clip,
                    self.eps_clip
                )
                value_loss_unclipped = self.MseLoss(state_values_new, mb_returns_flat)
                value_loss_clipped = self.MseLoss(value_pred_clipped, mb_returns_flat)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                
                # Entropy
                entropy_loss = dist_entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping if requested
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                loss_item = loss.item()

            # Optionally early stop on KL
            if self.target_kl is not None:
                # approximate KL: mean(old_logprob - new_logprob)
                with torch.no_grad():
                    new_logprobs, _, _ = self.policy.evaluate(old_states, old_actions, self.device)
                    approx_kl = (old_logprobs - new_logprobs).mean().item()
                if approx_kl > self.target_kl:
                    # print(f"Early stopping at epoch {epoch} due to KL {approx_kl:.5f} > target {self.target_kl}")
                    break

        # Sync old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear_memory()

        return loss_item