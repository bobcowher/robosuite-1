import robosuite as suite
import numpy as np
import os
import torch
from agent import Agent

# Set up device as either the GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Loading device: {device}")

# Setup the necessary paths for training
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./plots"):
    os.makedirs("./plots")

env_name = "Lift"

# Create the environment
env = suite.make(
    env_name,                        # Environment
    robots=["Panda"],           # Use two Panda robots
    controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
    has_renderer=True,                   # Enable rendering
    render_camera="sideview",           # Camera view
    has_offscreen_renderer=True,        # No offscreen rendering
    control_freq=20,                     # Control frequency
)

# Reset the environment and get environment params
obs = env.reset()

print(obs)

print(torch.tensor(obs, dtype=torch.float).shape())

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = Agent(state_dim, action_dim, max_action, batch_size=16, policy_freq=2,
            discount=0.99, device=device, tau=0.005, policy_noise=0.2, expl_noise=0.1,
            noise_clip=0.5, start_timesteps=1e4, learning_rate=0.0001, env_name=env_name, lr_decay_factor=0.999)


stats = agent.train(env, max_timesteps=2e7, batch_identifier=0)

# # Display the environment for a few seconds before executing any actions
# for _ in range(100):
#     env.render()
#
# # Execute actions in the environment
# for i in range(1000):
#     # Each robot has 7 DOF, so we generate random actions for each robot's joints
#     action = np.random.randn(8) * 0.1
#
#     print(action)
#
#     # Apply the action to the environment
#     obs, reward, done, info = env.step(action)
#     print(reward)
#
#     # Render the environment
#     env.render()
#
# env.close()