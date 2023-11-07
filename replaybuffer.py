import numpy as np
import pickle
import os

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.iteration = 0
        self.save_dir = "saved_buffers"

        self.load_from_disk()

    def add(self, transition):
        if len(self.storage) < self.max_size:
            self.storage.append(transition)
        else:
            self.storage.remove(self.storage[0])
            self.storage.append(transition)

        if self.iteration % 1000 == 0:
            self.save_to_disk()

    def save_to_disk(self, save_file="latest"):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Define the path to save the buffer
        save_path = os.path.join(self.save_dir, f"{save_file}.pkl")

        # Save the ReplayBuffer object to the file
        with open(save_path, 'wb') as f:
            pickle.dump(self.storage, f)

    def load_from_disk(self, filename='latest'):
        latest_path = os.path.join(self.save_dir, f"{filename}.pkl")


        if os.path.exists(latest_path):
            try:
                print(f"Attempting to load replay buffer from {latest_path}")
                with open(latest_path, 'rb') as f:
                    self.storage = pickle.load(f)

                print(f"Loaded ReplayBuffer from {latest_path}")
                print(f"Current buffer size is {len(self.storage)}")

            except:
                print(f"Failed to load pickle file from {latest_path}")



    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        batch_states = np.array(batch_states)
        batch_next_states = np.array(batch_next_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards).reshape(-1, 1)
        batch_dones = np.array(batch_dones)

        return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones

    def can_sample(self, batch_size):
        if len(self.storage) > batch_size * 10:
            return True
        else:
            return False