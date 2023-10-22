import robosuite as suite
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# Create the environment
env = suite.make(
    "TwoArmLift",                        # Environment
    robots=["Panda", "Panda"],           # Use two Panda robots
    controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
    has_renderer=True,                   # Enable rendering
    render_camera="frontview",           # Camera view
    has_offscreen_renderer=True,        # No offscreen rendering
    control_freq=20,                     # Control frequency
)

# Initial reset of the environment
obs = env.reset()

# Display the environment for a few seconds before executing any actions
for _ in range(100):
    env.render()

# Execute actions in the environment
for i in range(1000):
    # Each robot has 7 DOF, so we generate random actions for each robot's joints
    action_robot1 = np.random.randn(8) * 0.1
    action_robot2 = np.random.randn(8) * 0.1
    action = np.concatenate([action_robot1, action_robot2])
    print(action)

    # Apply the action to the environment
    obs, reward, done, info = env.step(action)
    print(reward)

    # Render the environment
    env.render()

env.close()