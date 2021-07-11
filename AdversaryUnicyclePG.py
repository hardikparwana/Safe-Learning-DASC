## Taken from

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse

import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser()
# parser.add_argument('--env', help='CartPole or LunarLander OpenAI gym environment', type=str)
# parser.add_argument('--use_cuda', help='Use if you want to use CUDA', action='store_true')


class Params:
    NUM_EPOCHS = 5000
    ALPHA = 5e-3        # learning rate
    BATCH_SIZE = 64     # how many episodes we want to pack into an epoch
    GAMMA = 0.99        # discount rate
    HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier


dt = 0.1

class follower:
    
    def __init__(self,X0,dt):
        self.X  = X0
        self.dt = dt
        
    def step(self,u,w):
        
        self.X = self.X + np.array([u*np.cos(self.X[2]),u*np.sin(self.X[2]),w])*dt
        
        return self.X
    
    def render(self,lines,areas,body):
        length = 5
        FoV = np.pi/3

        x = np.array([self.X[0],self.X[1]])
        theta = self.X[2]
        theta1 = theta + FoV/2
        theta2 = theta - FoV/2
        e1 = np.array([np.cos(theta1),np.sin(theta1)])
        e2 = np.array([np.cos(theta2),np.sin(theta2)])

        P1 = x + length*e1
        P2 = x + length*e2  

        triangle_hx = [x[0] , P1[0], P2[0], x[0] ]
        triangle_hy = [x[1] , P1[1], P2[1], x[1] ]
        triangle_v = [ x,P1,P2,x ]  

        lines.set_data(triangle_hx,triangle_hy)
        areas.set_xy(triangle_v)

        length2 = 3

        # scatter plot update
        body.set_offsets([x[0],x[1]])
#         sc.set_offsets(np.c_[x,y])

        return lines, areas, body
       
class target:
    
    def __init__(self,X0,dt):
        self.X = X0
        self.dt = dt
        self.t0 = 0
        self.speed = 0
        self.theta = 0
        
    def step(self,a,alpha):
        
        if (self.speed<2):
            self.speed = self.speed + a*self.dt
            
        self.theta = self.theta + alpha*dt
        
        if self.theta>np.pi:
            self.theta = self.theta - 2*np.pi
        if self.theta<-np.pi:
            self.theta = self.theta + 2*np.pi
        
        self.X = self.X + np.array([ self.speed*np.cos(self.theta),self.speed*np.sin(self.theta) ])*dt
        return self.X
    
    def render(self,body):
        length = 3
        FoV = np.pi/3

        x = np.array([self.X[0],self.X[1]])

        # scatter plot update
        body.set_offsets([x[0],x[1]])
#         sc.set_offsets(np.c_[x,y])

        return body
    
def wrap_angle(angle):
    if angle>np.pi:
        angle = angle - 2*np.pi
    if angle<-np.pi:
        angle = angle + 2*np.pi
    return angle
    
def compute_reward(F_X,T_X):
    
    FoV = 30*np.pi/180
    max_D = 3
    min_D = 0.7
    beta = np.arctan2(T_X[1]-F_X[1],T_X[0]-F_X[0])
    
    angle_diff = wrap_angle(beta - F_X[2])
    
    distance = np.sqrt( (T_X[0]-F_X[0])**2 + (T_X[1]-F_X[1])**2 )
    
    if np.abs(angle_diff)>FoV:
        reward_angle = -1
    else:
        reward_angle = np.abs(FoV-angle_diff)/FoV
    
    if distance>max_D:
        reward_distance = -1
    elif distance<min_D:
        reward_distance = -1
    else:
        # reward_distance = np.abs(distance-min_D)*np.abs(distance-max_D)*4/(max_D-min_D)**2
        mean_D = (max_D-min_D)/2
        c = mean_D**2
        reward_distance = c - (distance - mean_D)**2
        
    reward = reward_angle*reward_distance
    
    return reward


# Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        # x = normalize(x, dim=1)
        x = self.net(x)
        return x

# class env_custom:
#     def __init__(self):

#         self.agentF = follower(np.array([0,0.2,0]),dt)
#         self.agentT = target(np.array([1,0]),dt)

#         self.observation_space

class PolicyGradient:
    def __init__(self, problem: str = "CartPole", use_cuda: bool = False):

        self.NUM_EPOCHS = Params.NUM_EPOCHS
        self.ALPHA = Params.ALPHA
        self.BATCH_SIZE = Params.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.HIDDEN_SIZE = Params.HIDDEN_SIZE
        self.BETA = Params.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.action_dim = 2
        self.state_dim = 5

        # instantiate the tensorboard writer
        self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'NH={self.HIDDEN_SIZE},'
                                            f'BETA={self.BETA}')

        # create the environment
        # self.env = gym.make('CartPole-v1') if problem == "CartPole" else gym.make('LunarLander-v2')

        # the agent driven by a neural network architecture
        self.agent = Agent(observation_space_size=5,
                           action_space_size=2,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)

        self.total_rewards = deque([], maxlen=100)

        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False

    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        episode = 0
        epoch = 0

        # init the epoch arrays
        # used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.action_dim), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while True:
            print(f"episode{episode}")
            if episode % 5 == 0:
                render_env = True
            else:
                render_env = False
            # play an episode of the environment
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             episode) = self.play_episode(episode=episode,render_env=True)

            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # if the epoch is over - we have epoch trajectories to perform the policy gradient
            if episode >= self.BATCH_SIZE:

                # reset the rendering flag
                self.finished_rendering_this_epoch = False

                # reset the episode count
                episode = 0

                # increment the epoch
                epoch += 1

                # calculate the loss
                loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                    weighted_log_probs=epoch_weighted_log_probs)

                # zero the gradient
                self.adam.zero_grad()

                # backprop
                loss.backward()

                # update the parameters
                self.adam.step()

                # feedback
                print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                      end="",
                      flush=True)

                self.writer.add_scalar(tag='Average Return over 100 episodes',
                                       scalar_value=np.mean(self.total_rewards),
                                       global_step=epoch)

                self.writer.add_scalar(tag='Entropy',
                                       scalar_value=entropy,
                                       global_step=epoch)

                # reset the epoch arrays
                # used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

                # check if solved
                if np.mean(self.total_rewards) > 200:
                    print('\nSolved!')
                    break

        # close the environment
        self.env.close()

        # close the writer
        self.writer.close()

    def play_episode(self, episode: int, render_env: bool):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        # state = self.env.reset()

        agentF = follower(np.array([0,0.2,0]),dt)
        agentT = target(np.array([1,0]),dt)

        T = agentT.X
        F = agentF.X    
        state = np.array([F[0],F[1],F[2],T[0],T[1]])

        # initialize the episode arrays
        # episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, 2), device=self.DEVICE)
        average_rewards = np.empty(shape=(0,), dtype=np.float)
        episode_rewards = np.empty(shape=(0,), dtype=np.float)

        t = 0
        # dt = 0.1

        if render_env:
            plt.ion()

            fig = plt.figure()
            ax = plt.axes(xlim=(0,20),ylim=(-10,10))

            lines, = ax.plot([],[],'o-')
            areas, = ax.fill([],[],'r',alpha=0.1)
            bodyF = ax.scatter([],[],c='r',s=10)
            
            bodyT = ax.scatter([],[],c='g',s=10)

        # episode loop
        while True:

            t += dt
       
            if t<1:
                uL = 0.2
                vL = 0.1
            elif t<3:
                uL = 0.2
                vL = 0
            else:
                uL = 0.2
                vL = 0.2

            # get the action logits from the agent - (preferences)
            # action_logits = self.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))
            # action_org = self.agent(torch.tensor(state).float())
            # append the logits to the episode logits list

            # sample an action according to the action distribution
            # action = Categorical(logits=action_logits).sample()
            action_logits = self.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))
            action = action_logits.detach()
            print(f"action:{action.cpu().numpy()}, action_logits:{action_logits}")

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state
            # state, reward, done, _ = self.env.step(action=action.cpu().item())
            action_numpy = action.cpu().numpy()
            u = action_numpy[0]
            v = action_numpy[1]
            T_ns = agentT.step(uL,vL)  #agentT.step(0.2,0.5)        
            F_ns = agentF.step(u,v)
            reward = compute_reward(F_ns,T_ns)
            state = np.array([F[0],F[1],F[2],T[0],T[1]])

            if render_env:
                lines, areas, bodyF = agentF.render(lines,areas,bodyF)
                bodyT = agentT.render(bodyT)
                
                fig.canvas.draw()
                fig.canvas.flush_events()

            if reward<0:
                done = True
            else:
                done = False

            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            average_rewards = np.concatenate((average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)

            # the episode is over
            if done:

                # increment the episode
                episode += 1

                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.GAMMA)
                discounted_rewards_to_go -= average_rewards  # baseline - state specific average

                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True

                return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.BETA * entropy

        return policy_loss + entropy_bonus, entropy

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """
            Calculates the sequence of discounted rewards-to-go.
            Args:
                rewards: the sequence of observed rewards
                gamma: the discount factor
            Returns:
                discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards


def main():
    # args = parser.parse_args()
    # env = args.env
    # use_cuda = args.use_cuda

    # assert(env in ['CartPole', 'LunarLander'])

    policy_gradient = PolicyGradient(problem='CartPole', use_cuda=True)
    policy_gradient.solve_environment()


if __name__ == "__main__":
    main()