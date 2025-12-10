"""
Simple example showing how to instantiate the CarRacing encoder + actor/critic,
and how to instantiate a simple MLP actor/critic for LunarLander.
This file is not a trainer; it's a usage snippet you can run to test shapes.
"""
import torch
from actor_critic_carracing import Actor, QNetwork, MLPActor, MLPCritic


def test_carracing_shapes():
    # batch of 4 RGB images 96x96
    imgs = torch.randint(0, 256, (4, 3, 96, 96), dtype=torch.uint8)
    actor = Actor(feature_dim=256, action_dim=3)
    qnet = QNetwork(feature_dim=256, action_dim=3)

    with torch.no_grad():
        mean, log_std = actor.forward(imgs)
        a, logp, z = actor.sample(imgs)
        q = qnet.forward(imgs, a)

    print("Actor mean shape:", mean.shape)   # (4,3)
    print("Actor log_std shape:", log_std.shape)
    print("Sampled action shape:", a.shape)
    print("Q shape:", q.shape)


def test_lunarlander_shapes():
    # batch of 8 states
    states = torch.randn(8, 8)
    mlp_actor = MLPActor(state_dim=8, action_dim=2)
    mlp_critic = MLPCritic(state_dim=8, action_dim=2)

    with torch.no_grad():
        mean = mlp_actor(states)
        q = mlp_critic(states, torch.randn(8,2))

    print("Lunar actor shape:", mean.shape)  # (8,2)
    print("Lunar Q shape:", q.shape)


if __name__ == '__main__':
    print("Testing CarRacing components...")
    test_carracing_shapes()
    print("\nTesting LunarLander components...")
    test_lunarlander_shapes()
