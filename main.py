import robosuite as suite
import numpy as np
import os
import torch
from agent import Agent
from robosuite_environment import RoboSuiteWrapper
import cv2

# Set up device as either the GPU or CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Loading device: {device}")

# Setup the necessary paths for training
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./models"):
    os.makedirs("./models")
if not os.path.exists("./models/incremental"):
    os.makedirs("./models/incremental")
if not os.path.exists("./plots"):
    os.makedirs("./plots")


import robosuite as suite
import numpy as np
import cv2

env_name = "Lift"

env = suite.make(
    env_name,
    robots=["Panda"],
    controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
    has_renderer=True,
    use_camera_obs=True,
    render_camera="sideview",
    reward_shaping=True,
    control_freq=20,
)

env = RoboSuiteWrapper(env)

# Reset the environment and get environment params
obs = env.reset()

env = RoboSuiteWrapper(env)

#
# cv2.imshow('Camera Observation', obs)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()


state_dim = obs.shape[0]
action_dim = env.action_dim
max_action = 1

print(state_dim)
print(action_dim)


# min_action, max_action = env.action_spec
#
# print(max_action)

# agent = Agent(state_dim, action_dim, max_action=max_action, batch_size=16, policy_freq=4,
#             discount=0.99, device=device, tau=0.005, policy_noise=0.3, expl_noise=0.5,
#             min_policy_noise=0.15, min_expl_noise=0.2, noise_decay_rate=0.999,
#             noise_clip=0.5, start_timesteps=5e3, env_name=env_name,
#             replay_buffer_max_size=1000000)

agent = Agent(state_dim, action_dim, max_action=max_action, batch_size=32, policy_freq=4,
            discount=0.99, device=device, tau=0.005, policy_noise=0.15, expl_noise=0.2,
            min_policy_noise=0.15, min_expl_noise=0.2,
            noise_clip=0.5, start_timesteps=5e3, env_name=env_name,
            replay_buffer_max_size=1000000)

# learning_rate = 0.001
learning_rate = 0.00025
learning_rate = 0.00001

# for i in range(10):
stats = agent.train(env, max_timesteps=1000000, batch_identifier=0, learning_rate=learning_rate)
    # learning_rate = learning_rate * .9

print("Completed first training iteration")

# agent = Agent(state_dim, action_dim, max_action=max_action, batch_size=32, policy_freq=4,
#             discount=0.99, device=device, tau=0.005, policy_noise=0.1, expl_noise=0.1,
#             noise_clip=0.5, start_timesteps=1e4, learning_rate=0.00005, env_name=env_name, lr_decay_factor=1.0,
#             replay_buffer_max_size=1000000)
#
# stats = agent.train(env, max_timesteps=100000, batch_identifier=4)
#
# print("Completed second training iteration")
#
# agent = Agent(state_dim, action_dim, max_action=max_action, batch_size=32, policy_freq=4,
#             discount=0.99, device=device, tau=0.005, policy_noise=0.1, expl_noise=0.1,
#             noise_clip=0.5, start_timesteps=1e4, learning_rate=0.00001, env_name=env_name, lr_decay_factor=1.0,
#             replay_buffer_max_size=1000000)
#
# stats = agent.train(env, max_timesteps=200000, batch_identifier=5)
#
# print("Completed third training iteration")