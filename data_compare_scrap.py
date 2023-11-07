import torch
from replaybuffer import ReplayBuffer

h_data = ReplayBuffer()
t_data = ReplayBuffer()

h_data.load_from_disk('human_feedback')
t_data.load_from_disk('latest')

h_batch = h_data.sample(1)
t_batch = t_data.sample(1)



for i in range(len(h_batch)):
    print(f"Comparing size of element {i}")
    print("Human: ", h_batch[i].shape)
    print("Training: ", t_batch[i].shape)