'''
---------------------------------------------------------
Implementing of TD3
---------------------------------------------------------
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import namedtuple, deque
import torch.optim as optim
import random
import matplotlib.pyplot as plt
#%matplotlib inline 

import os
import time


class follower:
    
    def __init__(self,X0,dt):
        self.X  = X0
        self.dt = dt
        
    def step(self,u,w):
        
        self.X = self.X + np.array([u*np.cos(self.X[2]),u*np.sin(self.X[2]),w])*dt
        
        self.X[2] = wrap_angle(self.X[2])

        return self.X
    
    def render(self,lines,areas,body):
        length = 3
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
        
        # self.X = self.X + np.array([ self.speed*np.cos(self.theta),self.speed*np.sin(self.theta) ])*dt
        self.X = self.X + np.array([a,alpha])*dt
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
        reward_angle = -1 #-np.abs(FoV-np.abs(angle_diff))/FoV  #-1
    else:
        reward_angle = np.abs(FoV-angle_diff)/FoV
    
    mean_D = (max_D-min_D)/2
    if distance>max_D:
        reward_distance = -1
    elif distance<min_D:
        reward_distance = -1
    else:
        # reward_distance = np.abs(distance-min_D)*np.abs(distance-max_D)*4/(max_D-min_D)**2
        mean_D = (max_D-min_D)/2
        c = mean_D**2
        reward_distance = c - (distance - mean_D)**2
    # reward_distance = (distance - min_D)*(max_D - distance)/(mean_D**2)
        
    reward = np.abs(reward_angle)*np.abs(reward_distance)
    if (reward_angle<0) or (reward_distance<0):
        reward = -reward
    
    return reward



#Construct Neural Networks

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc_units=256, fc1_units=256):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x)) * self.max_action

# Q1-Q2-Critic Neural Network

class Critic_Q(nn.Module):
    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256):
        super(Critic_Q, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc1_units)
        self.l2 = nn.Linear(fc1_units, fc2_units)
        self.l3 = nn.Linear(fc2_units, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc1_units)
        self.l5 = nn.Linear(fc1_units, fc2_units)
        self.l6 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xa = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(xa))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xa))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2
upper_bound = 3
lower_bound = -3

dt = 0.1

class TD3:
    def __init__(
        self,name,env,
        load = False,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4,
        lr_critic = 3e-4,
        batch_size = 100,
        buffer_capacity = 1000000,
        tau = 0.02,  #soft update parameter
        random_seed = 666,
        cuda = True,
        policy_noise=0.2, 
        std_noise = 0.1,
        noise_clip=0.5,
        policy_freq=2, #target network update period
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.env = env
        self.upper_bound = float(3)#float(self.env.action_space.high[0]) #action space upper bound
        self.lower_bound = float(-3)#float(self.env.action_space.low[0])  #action space lower bound
        self.create_actor()
        self.create_critic()
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen = buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.name = name
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise   
        
    def create_actor(self):
        params = {
            'state_size':      5,
            'action_size':     2,
            'max_action':      self.upper_bound
        }
        self.actor = Actor(**params).to(self.device)
        self.actor_target = Actor(**params).to(self.device)

    def create_critic(self):
        params = {
            'state_size':      5,
            'action_size':     2
        }
        self.critic = Critic_Q(**params).to(self.device)
        self.critic_target = Critic_Q(**params).to(self.device)

    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def add_to_replay_memory(self, state, action, reward, next_state, done):
      #add samples to replay memory
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self):
      #random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def learn_and_update_weights_by_replay(self,training_iterations):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        if len(self.replay_memory_buffer) < self.batch_size:
            return
        for it in range(training_iterations):
            mini_batch = self.get_random_sample_from_replay_mem()
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)

            # Training and updating Actor & Critic networks.
            #Train Critic
            target_actions = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise).to(self.device)

            #clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            #Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions)
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            #Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            # Optimize the critic
            self.crt_opt.zero_grad()
            critic_loss.backward()
            self.crt_opt.step()

            #Train Actor
            # Delayed policy updates
            if it % self.policy_freq == 0:
                # Minimize the loss
                actions = self.actor(state_batch)
                actor_loss,_ = self.critic(state_batch, actions)
                actor_loss = - actor_loss.mean()

                # Optimize the actor               
                self.act_opt.zero_grad()
                actor_loss.backward()
                self.act_opt.step()

                #Soft update target models
                self.soft_update_target(self.critic, self.critic_target)
                self.soft_update_target(self.actor, self.actor_target)

    def soft_update_target(self,local_model,target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def policy(self,state):
        """select action based on ACTOR"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        return np.squeeze(actions)



def eval_policy(agent, seed, eval_episodes=10):
    # eval_env = eval_env
    avg_reward = 0.

    plt.ion()

    fig = plt.figure()
    ax = plt.axes(xlim=(0,20),ylim=(-10,10))

    lines, = ax.plot([],[],'o-')
    areas, = ax.fill([],[],'r',alpha=0.1)
    bodyF = ax.scatter([],[],c='r',s=10)
    
    bodyT = ax.scatter([],[],c='g',s=10)


    for _ in range(eval_episodes):
        # state, done = eval_env.reset(), False
        agentF = follower(np.array([0,0.2,0]),dt)
        agentT = target(np.array([1,0]),dt)
        done = False

        T = agentT.X
        F = agentF.X    
        state = np.array([F[0],F[1],F[2],T[0],T[1]])

        t = 0

        while not done:
            action = agent.policy(np.array(state))

            T = agentT.X
            F = agentF.X    
            # print(T)
            #print(F)
            state = np.array([F[0],F[1],F[2],T[0],T[1]])
            # state, reward, done, _ = eval_env.step(action)

            u = action[0]
            v = action[1]

            uL = 0.1
            vL = 0.2*np.sin(np.pi*t/5) #  0.1
            # print(f"t:{t}, vL:{vL}")
            
            T_ns = agentT.step(uL,vL)        
            F_ns = agentF.step(u,v)
            reward = compute_reward(F_ns,T_ns)
            
            #print(reward)
            
            lines, areas, bodyF = agentF.render(lines,areas,bodyF)
            bodyT = agentT.render(bodyT)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if reward<0:
                done = True

            avg_reward += reward

            t += dt

    avg_reward /= eval_episodes

    plt.close()

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

"""Training the agent"""
gym.logger.set_level(40)



def train(args):
    env = gym.make('BipedalWalker-v3')
    agent = TD3(args.rl_name, env, gamma=args.gamma, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                batch_size = args.batch_size, buffer_capacity=args.buffer_capacity, tau=args.tau, random_seed=args.seed,
                policy_freq=args.policy_freq)
    time_start = time.time()        # Init start time
    ep_reward_list = []
    avg_reward_list = []
    total_timesteps = 0

    epsilon = 0.5
    epsilon_decay = 0.99

    epsilon_min = 0.1

    timestep_list = []
    avg_timestep_list = []

    for ep in range(args.total_episodes):
        # state = env.reset()
        episodic_reward = 0
        timestep = 0

        agentF = follower(np.array([0,0.2,0]),dt)
        agentT = target(np.array([1,0]),dt)

        T = agentT.X
        F = agentF.X    
        state = np.array([F[0],F[1],F[2],T[0],T[1]])

        t = 0

        uL = 0.1
        vL = 0.2*np.sin(np.pi*t) #  0.1

        epsilon = epsilon*epsilon_decay
        if epsilon<epsilon_min:
            epsilon = epsilon_min
        
        for st in range(args.max_steps):
            # Select action randomly or according to policy
            # if total_timesteps < args.start_timestep:
            if (np.random.random()<epsilon):
                u = -3 + 6*random.uniform(0,1)
                v = -3 + 6*random.uniform(0,1)
                action = [u,v]
            else:
                action = agent.policy(state)

            # action = agent.policy(state)

            # Recieve state and reward from environment.
            # next_state, reward, done, info = env.step(action)
            u = action[0]
            v = action[1]

            uL = 0.1
            vL = 0.5*np.sin(np.pi*t/5) #  0.1


            T_ns = agentT.step(uL,vL)  #agentT.step(0.2,0.5)        
            F_ns = agentF.step(u,v)
            reward = compute_reward(F_ns,T_ns)
            next_state = np.array([F[0],F[1],F[2],T[0],T[1]])

            if reward<0:
                done = True
            else:
                done = False
            # done = False
            # print("target state",T_ns)

            #change original reward from -100 to -5 and 5*reward for other values
            episodic_reward += reward
            # if reward == -100:
            #     reward = -5
            # else:
            #     reward = 5 * reward
            agent.add_to_replay_memory(state, action, reward, next_state, done)          
            # End this episode when `done` is True
            if done:
                break
            state = next_state
            timestep += 1     
            total_timesteps += 1

            t += dt

        ep_reward_list.append(episodic_reward)
        # Mean of last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)

        timestep_list.append(timestep)

        avg_timestep = np.mean(timestep_list[-100:])
        avg_timestep_list.append(avg_timestep)

        # if avg_reward > 290:
        if ep % 50 == 0:
            test_reward = eval_policy(agent, seed=88, eval_episodes=1)
            if test_reward > 300:
                final_test_reward = eval_policy(agent, seed=88, eval_episodes=10)
                if final_test_reward > 300:
                    print("===========================")
                    print('Task Solved')
                    print("===========================")
                    #save weights
                    torch.save(agent.actor.state_dict(), 'actor.pth')
                    break            
        s = (int)(time.time() - time_start)
        agent.learn_and_update_weights_by_replay(timestep)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Episode Reward: {:.2f}, Moving Avg.Reward: {:.2f}, Fail Time: {:.2f} , epsilon: {:.2f}, Time: {:02}:{:02}:{:02}'
                .format(ep, total_timesteps, timestep,
                      episodic_reward, avg_reward, avg_timestep, epsilon, s//3600, s%3600//60, s%60)) 
    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Epsiodic Reward")
    plt.show()

    plt.plot(timestep_list)
    plt.xlabel("Episode")
    plt.ylabel("Time to Failure")
    plt.show()

    env.close()

import argparse
parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="BipedalWalkerHardcore-v3")
parser.add_argument('--rl-name', default="td3")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',help='target smoothing coefficient(τ)')
parser.add_argument('--lr-actor', type=float, default=0.0003, metavar='G',help='learning rate of actor')
parser.add_argument('--lr-critic', type=float, default=0.0003, metavar='G',help='learning rate of critic')
parser.add_argument('--seed', type=int, default=123456, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='batch size (default: 256)')
parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='N', help='buffer_capacity')
parser.add_argument('--max-steps', type=int, default=10000, metavar='N',help='maximum number of steps of each episode')
parser.add_argument('--total-episodes', type=int, default=3000, metavar='N',help='total training episodes')
parser.add_argument('--policy-freq', type=int, default=2, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
args = parser.parse_args("")

train(args)