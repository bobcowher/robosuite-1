import robosuite as suite
import numpy as np
import torch

class RoboSuiteWrapper(suite.environments.manipulation.lift.Lift):

    def __init__(self, env_name, test=False):
        super().__init__(RoboSuiteWrapper, env_name, robots=["Panda"], cont)
        if test == False:
            self.env = suite.make(
                env_name,  # Environment
                robots=["Panda"],  # Use two Panda robots
                controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
                has_renderer=True,  # Enable rendering
                # render_camera="sideview",           # Camera view
                # has_offscreen_renderer=True,        # No offscreen rendering
                control_freq=20,  # Control frequency
            )
            print(type(self.env))
        else:
            print("Running in test mode.")
            self.env = suite.make(
                env_name,  # Environment
                robots=["Panda"],  # Use two Panda robots
                controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
                has_renderer=True,  # Enable rendering
                render_camera="sideview",           # Camera view
                has_offscreen_renderer=True,        # No offscreen rendering
                control_freq=20,  # Control frequency
            )
            print(type(self.env))


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.observation_to_tensor(observation)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self.observation_to_tensor(observation)
        return observation


    def observation_to_tensor(self, obs):
        # Convert each array in the ordered dictionary to a flattened numpy array
        flattened_arrays = [np.array(item).flatten() for item in obs.values()]

        # Concatenate all the flattened arrays to get a single array
        concatenated_array = np.concatenate(flattened_arrays)

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(concatenated_array, dtype=torch.float32)