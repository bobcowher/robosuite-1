from model import Actor
from model import Critic
import torch
import torch.nn.functional as F
from replaybuffer import ReplayBuffer
import numpy as np
import time
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Agent(object):

    def __init__(self, state_dim, action_dim, max_action, batch_size, policy_freq, discount, tau=0.005, eval_freq=100,
                 policy_noise=0.2, min_policy_noise=0.1, expl_noise=0.1, min_expl_noise=0.1, noise_decay_rate=0.999, noise_clip=0.5, start_timesteps=1e4, device=None, env_name=None,
                 replay_buffer_max_size=1000000, lr_decay_factor=1, min_learning_rate=0.00001, decay_step=1000):
        """

        :param state_dim:
        :param action_dim:
        :param max_action:
        :param batch_size: Size of the sample batch used
        :param policy_freq: Number of iterations to wait before the policy network (Actor model) is updated
        :param device: Device the model should train on (cpu v.s gpu)
        :param discount: The discount factor for future rewards
        :param eval_freq: How often we should evaluate the model
        :param tau: target model update rate
        :param start_timesteps: Number of timesteps to take in warmup mode
        """
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = None

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = None

        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_optimizer = None

        self.max_action = max_action
        self.action_dim = action_dim

        # Loading the first critic model.
        critic_model_loaded = self.critic1.load_the_model(weights_filename=f"{env_name}_critic1_latest.pt")
        self.critic1_target.load_the_model(weights_filename=f"{env_name}_critic1_latest.pt")

        # Loading the second critic model
        self.critic2.load_the_model(weights_filename=f"{env_name}_critic2_latest.pt")
        self.critic2_target.load_the_model(weights_filename=f"{env_name}_critic2_latest.pt")

        # Loading the actor model
        actor_model_loaded = self.actor.load_the_model(weights_filename=f"{env_name}_actor_latest.pt")
        self.actor_target.load_the_model(weights_filename=f"{env_name}_actor_latest.pt")


        self.batch_size = batch_size
        self.policy_freq = policy_freq
        self.discount = discount
        self.eval_freq = eval_freq
        self.tau = tau
        self.policy_noise = policy_noise
        self.min_policy_noise = min_policy_noise
        self.expl_noise = expl_noise
        self.min_expl_noise = min_expl_noise
        self.noise_decay_rate = noise_decay_rate
        self.noise_clip = noise_clip
        self.env_name = env_name
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size)
        self.decay_step = decay_step
        self.lr_decay_factor = lr_decay_factor

        # self.start_timesteps = start_timesteps
        if critic_model_loaded and actor_model_loaded:
            self.start_timesteps = 0
            print(f"Model successfully loaded. Setting startup timesteps to 0")
        else:
            self.start_timesteps = start_timesteps
            print(f"No model loaded. Setting startup timesteps to {start_timesteps}")

        print(f"Configured agent with device: {self.device}")



    def select_action(self, state):
        # state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        # if not isinstance(state, torch.Tensor):
        #     print(state)
        #     exit

        if len(state.shape) == 3:
            # Add a batch dimension
            state = state.unsqueeze(0)

        state = state.to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()



    def train(self, env, max_timesteps, actor_learning_rate, critic_learning_rate, weight_decay, batch_identifier):
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_learning_rate, weight_decay=weight_decay)
        self.critic1_optimizer = torch.optim.AdamW(self.critic1.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)
        self.critic2_optimizer = torch.optim.AdamW(self.critic1.parameters(), lr=critic_learning_rate, weight_decay=weight_decay)



        stats = {'Returns': [], 'AvgReturns': []}

        # evaluations = [self.evaluate_policy(env)]

        writer = SummaryWriter(log_dir=f"./tensorboard_logdir/{self.env_name}/{datetime.now().strftime('%Y-%m-%d')}")
        #
        # # This section is just to add model visualization.
        # dummy_input = torch.randn(1, 84)
        # dummy_action = torch.randn(1, 8)
        #
        # dummy_input = dummy_input.to(self.device)
        # dummy_action = dummy_action.to(self.device)
        #
        # writer.add_graph(self.actor, dummy_input)
        # writer.add_graph(self.critic, (dummy_input, dummy_action))

        total_timesteps = 0
        timesteps_since_eval = 0
        best_episode_reward = 0
        episode_num = 0
        done = True
        t0 = time.time()

        while total_timesteps < max_timesteps:

            actor_learning_rate = self.actor_optimizer.param_groups[0]["lr"]
            critic1_learning_rate = self.critic1_optimizer.param_groups[0]["lr"]

            # if total_timesteps % self.decay_step == 0:
            #     self.adjust_learning_rate(total_timesteps=total_timesteps)

            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print(f"Total Timesteps: {total_timesteps} Episode Num: {episode_num} Reward: {episode_reward} "
                          f"Learning Rate: {critic1_learning_rate:.10f}:{actor_learning_rate:.10f} Batch: {batch_identifier}")

                    actor_loss, critic_loss = self.learn(replay_buffer=self.replay_buffer, epochs=10)
                    # writer.add_scalar(f'{self.env_name} - Learning Rate: {batch_identifier}', actor_learning_rate, total_timesteps)
                    writer.add_scalar(f'{self.env_name} - Explore Noise: {batch_identifier}', self.expl_noise, total_timesteps)
                    writer.add_scalar(f'{self.env_name} - Policy Noise: {batch_identifier}', self.policy_noise, total_timesteps)
                    writer.add_scalar(f'{self.env_name} - Returns: {batch_identifier}', episode_reward, total_timesteps)
                    writer.add_scalar(f'{self.env_name} - Actor Loss: {batch_identifier}', actor_loss, total_timesteps)
                    writer.add_scalar(f'{self.env_name} - Critic Loss: {batch_identifier}', critic_loss, total_timesteps)


                    # writer.add_scalar(f'{self.env_name} - Returns Per Step: {batch_identifier}', (episode_reward / episode_timesteps), total_timesteps)

                    if episode_reward > best_episode_reward:
                        best_episode_reward = episode_reward
                        self.critic1.save_the_model(weights_filename='critic_best.pt')
                        self.actor.save_the_model(weights_filename='actor_best.pt')

                # When the training step is done, we reset the state of the environment
                obs = env.reset()

                writer.flush()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                if episode_num % 10 == 0:
                    self.save()

                if self.policy_noise > self.min_policy_noise:
                    self.policy_noise = self.policy_noise * self.noise_decay_rate

                if self.expl_noise > self.min_expl_noise:
                    self.expl_noise = self.expl_noise * self.noise_decay_rate



            # Before 10000 timesteps, we play random actions
            if total_timesteps < self.start_timesteps:
                action = np.random.randn(8) * 0.1
            # elif 0 <= total_timesteps % 500 <= 50:
            #     action = np.random.randn(8) * 0.1
            else:  # After 10000 timesteps, we switch to the model
                action = self.select_action(obs)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                        0, 1)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = env.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == env.max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward

            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            self.replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1

            total_timesteps += 1
            timesteps_since_eval += 1

            if total_timesteps % 10000 == 0:
                self.save(iteration=total_timesteps, folder='incremental/')





        return True

    def test(self, env, max_timesteps):

        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

        # print(f"Printing actor model")
        # self.actor.print_model()
        # print(f"Printing critic model")
        # self.actor.print_model()
        # print(f"Sleeping for 30 seconds to view...")
        # time.sleep(30)

        total_timesteps = 0
        episode_num = 0
        done = True
        t0 = time.time()

        while total_timesteps < max_timesteps:
            time.sleep(0.07) # Slow down enough to see the environment run.
            env.render()
            # If the episode is done

            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))

                # When the training step is done, we reset the state of the environment
                obs = env.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1


            action = self.select_action(obs)

            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            if self.expl_noise != 0:
                action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(0, 1)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            obs, reward, done, _ = env.step(action)

            print(f"Step reward was {reward}")

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == env.max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward


            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            episode_timesteps += 1
            total_timesteps += 1

    def learn(self, replay_buffer: ReplayBuffer, epochs):
        average_critic_loss_list = []
        average_actor_loss_list = []

        # If batch size isn't large enough to sample, exit here.
        if not replay_buffer.can_sample(self.batch_size):
            return 0, 0

        for epoch in range(epochs):
            if replay_buffer.can_sample(self.batch_size):
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size=self.batch_size)

                state = torch.Tensor(batch_states).to(self.device)
                next_state = torch.Tensor(batch_next_states).to(self.device)
                action = torch.Tensor(batch_actions).to(self.device)
                reward = torch.Tensor(batch_rewards).to(self.device)
                done = torch.Tensor(batch_dones).to(self.device)

                # Step 5: From the next state s', the actor target plays the next action a'
                next_action = self.actor_target(next_state).to(self.device)

                # Step 6: Add Gaussian noise
                noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(self.device)
                noise = noise.clamp(-self.noise_clip, +self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Step 7: Get critic q value
                target_q1 = self.critic1_target(next_state, next_action)
                target_q2 = self.critic2_target(next_state, next_action)

                # # Step 8: We keep the minimum of these two Q-values
                target_q = torch.min(target_q1, target_q2)

                # Step 9: We get the final target of the two Critic models, which is Qt = r + y * min(Qt1, Qt2), where y is the discount factor.
                target_q = reward + ((1 - done) * self.discount * target_q).detach()

                # Step 10: The two critic models should take each the couple (s, a) as input and return two Q-Values(Q1 of s,a and Q2 of s,a)
                current_q1 = self.critic1(state, action)
                current_q2 = self.critic2(state, action)

                # print("Target Q: ", target_q)
                # print("Current Q: ", current_q)


                # Compute critic loss, complete backprop, and clip gradients
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

                self.critic1_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()

                critic_loss.backward()

                self.critic1_optimizer.step()
                self.critic2_optimizer.step()


                # Compute actor loss, complete backprop, and clip gradients
                actor_loss = -self.critic1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # update the target models and clip gradients.
                for target_param, main_param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

                for target_param, main_param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

                if epoch % self.policy_freq == 0:
                    max_grad_norm = 1.0

                    # Use Polyak averaging to update the target weights
                    for target_param, main_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                        target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

                # Append loss values for later calculation of averages
                average_critic_loss_list.append(critic_loss.item())
                average_actor_loss_list.append(actor_loss.item())

        average_critic_loss = sum(average_critic_loss_list) / len(average_critic_loss_list)
        average_actor_loss = sum(average_actor_loss_list) / len(average_actor_loss_list)

        return average_actor_loss, average_critic_loss


    def save(self, iteration="latest", folder=""):
        self.actor.save_the_model(weights_filename=f"{folder}{self.env_name}_actor_{iteration}.pt")
        self.critic1.save_the_model(weights_filename=f"{folder}{self.env_name}_critic1_{iteration}.pt")
        self.critic2.save_the_model(weights_filename=f"{folder}{self.env_name}_critic2_{iteration}.pt")