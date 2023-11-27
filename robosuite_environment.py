import cv2
import robosuite as suite
import numpy as np
import torch

from robosuite.wrappers import Wrapper

from PIL import Image


class RoboSuiteWrapper(Wrapper):

    def __init__(self, env):

        super().__init__(env)

        self.max_episode_steps = 300
        self.current_episode_step = 0
        self.image_height = 256
        self.image_width = 256

        # if not test:

        # else:
        #     self.env = suite.make(
        #         env_name,  # Environment
        #         robots=["Panda"],  # Use two Panda robots
        #         controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),  # Controller
        #         has_renderer=True,  # Enable rendering
        #         render_camera="sideview",           # Camera view
        #         has_offscreen_renderer=True,        # No offscreen rendering
        #         control_freq=20,  # Control frequency
        #     )



    def step(self, action):
        _, reward, done, info = super().step(action)
        observation = self.get_observation()


        # Increment timesteps and set done if max timesteps reached
        self.current_episode_step += 1

        if self.current_episode_step == self.max_episode_steps:
            done = True

        return observation, reward, done, info


    def get_observation(self):
        observation = self.sim.render(width=256, height=256, camera_name="frontview")

        # Reshape from a flat tensor to an image.
        observation = observation.reshape((self.image_height, self.image_width, 3))

        # Flip right side up
        observation = np.flipud(observation)

        # Convert to black and white
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        observation = observation / 255.0

        observation = torch.from_numpy(observation).float()

        observation = observation.unsqueeze(0)
        # observation = observation.unsqueeze(0)

        # img = np.array(img)
        # img = torch.from_numpy(img)
        # img = img.unsqueeze(0)
        # img = img.unsqueeze(0)
        # img = img / 255.0
        #
        # img = img.to(self.device)

        return observation


    def reset(self):
        self.current_episode_step = 0
        _ = super().reset()
        observation = self.get_observation()
        return observation


    def observation_to_tensor(self, obs):
        # Convert each array in the ordered dictionary to a flattened numpy array
        flattened_arrays = [np.array(item).flatten() for item in obs.values()]

        # Concatenate all the flattened arrays to get a single array
        concatenated_array = np.concatenate(flattened_arrays)

        # Convert the numpy array to a PyTorch tensor
        return torch.tensor(concatenated_array, dtype=torch.float32)
