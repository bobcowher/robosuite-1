import robosuite as suite
import numpy as np
import torch
from robosuite_environment import RoboSuiteWrapper


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# Create the environment
# env = suite.make(
#     "Lift",  # Environment
#     robots=["Panda"],  # Use two Panda robots
#     controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
#     has_renderer=True,  # Enable rendering
#     render_camera="sideview",  # Camera view
#     has_offscreen_renderer=True,  # No offscreen rendering
#     control_freq=20,  # Control frequency
# )

env = RoboSuiteWrapper("Lift", test=True)

# Initial reset of the environment
obs = env.reset()

# Display the environment for a few seconds before executing any actions
for _ in range(100):
    env.render()

# Execute actions in the environment
for i in range(1000):
    # Each robot has 7 DOF, so we generate random actions for each robot's joints
    action = np.random.randn(8) * 0.1

    # Apply the action to the environment
    obs, reward, done, info = env.step(action)
    print(reward)

    # Render the environment
    env.render()

env.close()