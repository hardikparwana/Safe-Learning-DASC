'''
---------------------------------------------------------
Implementing of TD3
---------------------------------------------------------
'''

from operator import is_
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from collections import namedtuple, deque
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *
#%matplotlib inline 

import os
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import cvxpy as cp

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3


def wrap_angle(angle):
    if angle>np.pi:
        angle = angle - 2*np.pi
    if angle<-np.pi:
        angle = angle + 2*np.pi
    return angle


def build_GP_model(N):
    GP_list = []
    noise = 0.01
    for i in range(N):
        kern = 1.0 * RBF(length_scale=2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))
        #kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
        GP_list.append(gp)
    return GP_list

@ignore_warnings(category=UserWarning)
@ignore_warnings(category=ConvergenceWarning)
def update_GP_dynamics(GP_list,X,Y, index, curr_X):
    # X, Y = filter_data(X,Y,curr_X)
    GP_list[index].fit(np.asarray(X).reshape(-1,1),np.asarray(Y).reshape(-1,1))


def filter_data(obs_x,obs_value,curr_x,thres=0.5):
    filter_x = []
    filter_value = []
    for index, t in enumerate(obs_x):
        if np.abs(t-curr_x)<thres:
            filter_x.append(t)
            filter_value.append(obs_value[index])
    return filter_x, filter_value

def predict_GP_dynamics(GP_list,N,X):
    Y = []
    Y_std = []
    for i in range(N):
        y_pred, sigma_pred = GP_list[i].predict(np.array([X]).reshape(-1, 1),return_std=True)
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def test_GP_dynamics(GP_list,X,index):
    Y = []
    Y_std = []
    for i in X:
        y_pred, sigma_pred = GP_list[index].predict(np.array([i]).reshape(-1, 1),return_std=True)
        Y.append(y_pred[0][0])
        Y_std.append(sigma_pred[0])
    return np.asarray(Y), np.asarray(Y_std)

def compute_CLF(X,Xd):
    V = (np.linalg.norm(X-Xd))*2
    return V

def CBF_loss(agent,target,u,w):

    alpha = 0.1 #1.0 does not work

    h1,dh1_dxA,dh1_dxB = agent.CBF1_loss(agent,target)
    h2,dh2_dxA,dh2_dxB = agent.CBF2_loss(agent,target)
    h3,dh3_dxA,dh3_dxB = agent.CBF3_loss(agent,target)

    U = np.array([u,w]).reshape(-1,1)
    
    h1_ = dh1_dxA @ agent.xdot(U) + dh1_dxB @ target.xdot(target.U) + alpha*h1
    h2_ = dh2_dxA @ agent.xdot(U) + dh2_dxB @ target.xdot(target.U) + alpha*h2
    h3_ = dh3_dxA @ agent.xdot(U) + dh3_dxB @ target.xdot(target.U) + alpha*h3

    if h1_ > 0 and h2_ > 0 and h3_ > 0:
        return True, h1_, h2_, h3_
    else:
        return False, h1_, h2_, h3_


def CBF_CLF_QP(agent,target):

    u, w = agent.nominal_controller(target)
    U = np.array([u,w]).reshape(-1,1)

    alpha = 0.1 #1.0 does not work
    k = 0.1

    h1,dh1_dxA,dh1_dxB = agent.CBF1_loss(target)
    h2,dh2_dxA,dh2_dxB = agent.CBF2_loss(target)
    h3,dh3_dxA,dh3_dxB = agent.CBF3_loss(target)

    V,dV_dxA,dV_dxB = agent.CLF_loss(target)

    x = cp.Variable((2,1))


    objective = cp.Minimize(cp.sum_squares(x-U))
    
    # CLF constraint
    const = [dV_dxA @ agent.xdot(x) + dV_dxB @ target.xdot(target.U) <= -k*V]

    # CBF1 constraint
    const += [dh1_dxA @ agent.xdot(x) + dh1_dxB @ target.xdot(target.U) >= -alpha*h1]
    const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha*h2]
    const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha*h3]

    problem = cp.Problem(objective,const)
    result = problem.solve()

    if problem.status == 'optimal':
        return True, x.value[0,0], x.value[1,0]
    else:
        print("SBF QP not solvable")
        return False, 0, 0



    # G = np.array([ [ VA[0] , VA[1], -1.0   ],
    #                [ -h1A[0], -h1A[1], 0.0 ] ,
    #                [ -h2A[0], -h2A[1], 0.0 ] ,
    #                [ -h3A[0], -h3A[1], 0.0 ]    
    #                 ])

    # h = np.array([-VB - k*V ,
    #              h1B + alpha*h1  ,
    #              h2B + alpha*h2  ,
    #              h3B + alpha*h3 ]).reshape(-1,1)

    # #Convert numpy arrays to cvx matrices to set up QP
    # G = matrix(G,tc='d')
    # h = matrix(h,tc='d')

    # # Cost matrices
    # P = matrix(np.diag([2., 2., 6.]), tc='d')
    # q = matrix(np.array([ -2*u, -2*w, 0 ]))

    # solvers.options['show_progress'] = False
    # sol = solvers.qp(P, q, G, h)
    # u_bar = sol['x']

    # if sol['status'] != 'optimal':
    #     return False, 0, 0

    #return True, u_bar[0], u_bar[1]



#Construct Neural Networks

class Actor(nn.Module):
    def __init__(self, state_size, action_size, max_action, fc_units=256, fc1_units=256):  # 256,256
        super(Actor, self).__init__()

        self.max_action = max_action
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
        x = F.relu(self.fc1(state))
        # print(f"state:{state}, x:{x}")
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
upper_bound = 10
lower_bound = -10

dt = 0.05

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
        plot_freq = 1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.env = env
        self.upper_bound = float(30)#float(self.env.action_space.high[0]) #action space upper bound
        self.lower_bound = float(-30)#float(self.env.action_space.low[0])  #action space lower bound
        self.create_actor()
        self.create_actor_temp()
        self.create_critic()
        self.lr_actor = lr_actor  #/2
        self.lr_actor_step = lr_actor/10  #/10
        self.act_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.crt_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.set_weights()
        self.replay_memory_buffer = deque(maxlen = buffer_capacity)
        self.replay_buffer = deque(maxlen = buffer_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.policy_freq = policy_freq
        self.gamma = gamma
        self.name = name
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.std_noise = std_noise   
        self.plot_freq = plot_freq
        
    def create_actor(self):
        params = {
            'state_size':      1,
            'action_size':     3,
            'max_action':      self.upper_bound
        }
        self.actor = Actor(**params).to(self.device)
        self.actor_target = Actor(**params).to(self.device)

    def create_actor_temp(self):
        params = {
            'state_size':      1,
            'action_size':     3,
            'max_action':      self.upper_bound
        }
        self.actor_temp = Actor(**params).to(self.device)
        self.actor_target_temp = Actor(**params).to(self.device)

    def create_critic(self):
        params = {
            'state_size':      1,
            'action_size':     3
        }
        self.critic = Critic_Q(**params).to(self.device)
        self.critic_target = Critic_Q(**params).to(self.device)

    def set_weights(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


    def add_to_replay_memory(self, state, action, reward, next_state, done, obs_d1, obs_d2, obs_d3):
      #add samples to replay memory
        self.replay_memory_buffer.append((state, action, reward, next_state, done, obs_d1, obs_d2, obs_d3))

    def filter_data(self, state, action, reward, next_state, done):
        # print("buffer : ",self.replay_memory_buffer)
        for element in self.replay_memory_buffer:
            if np.abs(element[0]-state)<0.01:
                # print("repeated ", element[0], element[0]-state)
                return False
        return True


    def get_random_sample_from_replay_mem(self,timesteps=None):
      #random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)

        # random_sample = self.replay_memory_buffer

        return random_sample


    # a large learning rate or too many iterations of learning is bad
    def learn_and_update_weights_by_replay(self,training_iterations,episodic_reward):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        # iter_count = 1

        # changed = False

        if len(self.replay_memory_buffer) < self.batch_size:
            return
        print("TRAINING")
        for it in range(min(1000,training_iterations)):
            # if iter_count>200:
            #     break;
            # iter_count += 1
            # print("iter count",iter_count)
            #NEW
            lr_actor = self.lr_actor

            mini_batch = self.get_random_sample_from_replay_mem()
            state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
            action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
            reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
            next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
            done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
            obs_d1_batch = torch.from_numpy(np.vstack([i[5] for i in mini_batch])).float().to(self.device)
            obs_d2_batch = torch.from_numpy(np.vstack([i[6] for i in mini_batch])).float().to(self.device)
            obs_d3_batch = torch.from_numpy(np.vstack([i[7] for i in mini_batch])).float().to(self.device)

            # Training and updating Actor & Critic networks.
            #Train Critic
            # print("TRAINING")
            target_actions = self.actor_target(next_state_batch)
            offset_noises = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise).to(self.device)

            #clip noise
            offset_noises = offset_noises.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + offset_noises).clamp(self.lower_bound, self.upper_bound)

            #Compute the target Q value
            Q_targets1, Q_targets2 = self.critic_target(next_state_batch, target_actions)
            # print("")
            Q_targets = torch.min(Q_targets1, Q_targets2)
            Q_targets = reward_batch + self.gamma * Q_targets * (1 - done_list)

            #Compute current Q estimates
            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, Q_targets.detach()) + F.mse_loss(current_Q2, Q_targets.detach())
            
            # # Optimize the critic
            # self.crt_opt.zero_grad()
            # critic_loss.backward()
            # self.crt_opt.step()

            #Train Actor
            # Delayed policy updates

            if it % self.policy_freq == 0:
                # Minimize the loss
                actions = self.actor(state_batch)
                actor_loss,_ = self.critic(state_batch, actions)
                actor_loss = - actor_loss.mean()

                test_reward = 0
                for item, value in enumerate(actions):
                    test_reward-= ( value[0]-obs_d1_batch[item] )**2 + ( value[1]-obs_d2_batch[item] )**2 + ( value[2]-obs_d3_batch[item] )**2
                test_reward *= 100

                improved = False
                max_iter = 100
                test_reward_prev = test_reward
                test_reward_prev_first = test_reward_prev
                # print("here 1")
                
                # copy from actor to actor_temp
                for target_param, local_param in zip(self.actor_temp.parameters(), self.actor.parameters()):
                    target_param.data.copy_(local_param.data)

                print("previous reward ", test_reward_prev)
                while not improved and lr_actor>-self.lr_actor: #-0.0003*5:
                    lr_actor = lr_actor - self.lr_actor_step
                    
                    for g in self.act_opt.param_groups:                        
                        g['lr'] = lr_actor

                    # Optimize the actor               
                    self.act_opt.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.act_opt.step()

                    test_reward_new = 0
                    actions_temp = self.actor(state_batch)
                    for item, value in enumerate(actions_temp):
                        test_reward_new -= ( value[0]-obs_d1_batch[item] )**2 + ( value[1]-obs_d2_batch[item] )**2 + ( value[2]-obs_d3_batch[item] )**2
                    test_reward_new *= 100

                    print(f" lr: {lr_actor}, new reward:{test_reward_new}")
                    # print(f"In TRAIN:{lr_actor}, test_reward_new:{test_reward_new}, test_reward_prev:{test_reward_prev}")
                    if (test_reward_new > test_reward_prev):
                        improved = True
                        changed = True
                        break
 
                    # copy from actor_temp to actor
                    for target_param, local_param in zip(self.actor.parameters(), self.actor_temp.parameters()):
                        target_param.data.copy_(local_param.data)
                
                # if test_reward_new<test_reward_prev_first-0.02:
                #     print("reward before",test_reward_prev_first)
                #     print("reward after",test_reward_new)
                #     exit()

            # # Minimize the loss
            # actions = self.actor(state_batch)
            # actor_loss,_ = self.critic(state_batch, actions)
            # actor_loss = - actor_loss.mean()

            # # Optimize the actor               
            # self.act_opt.zero_grad()
            # actor_loss.backward()
            # self.act_opt.step()

            #Soft update target models
            self.soft_update_target(self.critic, self.critic_target)
            self.soft_update_target(self.actor, self.actor_target)

        # if changed==True:
        #     print("policy changed")

    def learn_and_update_weights_by_passed_replay(self ):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """

        iter_count = 1

        changed = False

        if len(self.replay_buffer) < self.batch_size:
            print("returning", self.batch_size)
            print(len(self.replay_buffer))
            return

        
       
        lr_actor = self.lr_actor

        mini_batch = self.replay_buffer
        state_batch = torch.from_numpy(np.vstack([i[0] for i in mini_batch])).float().to(self.device)
        action_batch = torch.from_numpy(np.vstack([i[1] for i in mini_batch])).float().to(self.device)
        reward_batch = torch.from_numpy(np.vstack([i[2] for i in mini_batch])).float().to(self.device)
        next_state_batch = torch.from_numpy(np.vstack([i[3] for i in mini_batch])).float().to(self.device)
        done_list = torch.from_numpy(np.vstack([i[4] for i in mini_batch]).astype(np.uint8)).float().to(self.device)
        obs_d1_batch = torch.from_numpy(np.vstack([i[5] for i in mini_batch])).float().to(self.device)
        obs_d2_batch = torch.from_numpy(np.vstack([i[6] for i in mini_batch])).float().to(self.device)
        obs_d3_batch = torch.from_numpy(np.vstack([i[7] for i in mini_batch])).float().to(self.device)


        # print("state batch", state_batch)
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

        actions = self.actor(state_batch)
        actor_loss,_ = self.critic(state_batch, actions)
        actor_loss = - actor_loss.mean()


        test_reward = 0
        for item, value in enumerate(actions):
            test_reward-= ( value[0]-obs_d1_batch[item] )**2 + ( value[1]-obs_d2_batch[item] )**2 + ( value[2]-obs_d3_batch[item] )**2
        test_reward *= 100

        improved = False
        max_iter = 100
        test_reward_prev = test_reward
        test_reward_prev_first = test_reward_prev
        # print("here 1")
        
        # copy from actor to actor_temp
        for target_param, local_param in zip(self.actor_temp.parameters(), self.actor.parameters()):
            target_param.data.copy_(local_param.data)

        print("previous reward ", test_reward_prev)
        while not improved and lr_actor>=0:#-self.lr_actor: #-0.0003*5:
            
            for g in self.act_opt.param_groups:                        
                g['lr'] = lr_actor

            # Optimize the actor               
            self.act_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.act_opt.step()

            test_reward_new = 0
            actions_temp = self.actor(state_batch)
            for item, value in enumerate(actions_temp):
                test_reward_new -= ( value[0]-obs_d1_batch[item] )**2 + ( value[1]-obs_d2_batch[item] )**2 + ( value[2]-obs_d3_batch[item] )**2
            test_reward_new *= 100

            print(f" lr: {lr_actor}, new reward:{test_reward_new}")
            # print(f"In TRAIN:{lr_actor}, test_reward_new:{test_reward_new}, test_reward_prev:{test_reward_prev}")
            if (test_reward_new > test_reward_prev):
                improved = True
                changed = True
                break

            # copy from actor_temp to actor
            for target_param, local_param in zip(self.actor.parameters(), self.actor_temp.parameters()):
                target_param.data.copy_(local_param.data)

            lr_actor = lr_actor - self.lr_actor_step

        

        # # Optimize the actor               
        # self.act_opt.zero_grad()
        # actor_loss.backward()
        # self.act_opt.step()

        #Soft update target models
        self.soft_update_target(self.critic, self.critic_target)
        self.soft_update_target(self.actor, self.actor_target)

        if changed==True:
            print("policy changed")

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



def eval_policy_sim(agent_TD3, seed, eval_episodes=10):
    # eval_env = eval_env
    avg_reward = 0.

    for _ in range(eval_episodes):
        # state, done = eval_env.reset(), False
        # agentF = follower(np.array([0,0.2,0]),dt)
        agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
        agentT = target(np.array([1,0]),dt)
        done = False

        t = 0

        T_ns_prev = agentT.X
        F_ns_prev = agentF.X

        TX = agentT.X
        FX = agentF.X    
        state = np.array([F[0],F[1],F[2],T[0],T[1]])

        while not done:
            action = agent_TD3.policy(np.array(state))

            # uL = 0.1
            # vL = 0.2*np.sin(np.pi*t/5) #  0.1

            uL = 0.5
            vL = 3.6*np.sin(np.pi*t) #  0.1 # 1.2

            T = agentT.X
            F = agentF.X    
            #print(T)
            #print(F)
            state = np.array([F[0],F[1],F[2],T[0],T[1]])
            # state, reward, done, _ = eval_env.step(action)

            u = action[0]
            v = action[1]
            
            T_ns = agentT.step(0.1,0.1)        
            F_ns = agentF.step(u,v)
            reward = compute_reward(F_ns,T_ns,F_ns_prev,T_ns_prev)
            

            if reward<0:
                done = True

            avg_reward += reward

            T_ns_prev = T_ns
            F_ns_prev = F_ns

            t += dt

    avg_reward /= eval_episodes

    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    return avg_reward


def compute_reward_model_error(e1,e2,e3):
    reward = e1**2 + e2**2 + e3**2
    # reward = e1 + e2 + e3

    # if reward<5:
    #     reward -= 5
    # elif reward>5:
    #     reward = 30
    return -100*reward

"""Training the agent"""
gym.logger.set_level(40)



def train(args):
    env = gym.make('BipedalWalker-v3')
    agent_TD3 = TD3(args.rl_name, env, gamma=args.gamma, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                batch_size = args.batch_size, buffer_capacity=args.buffer_capacity, tau=args.tau, random_seed=args.seed,
                policy_freq=args.policy_freq, plot_freq=args.plot_freq)
    time_start = time.time()        # Init start time
    ep_reward_list = []
    avg_reward_list = []
    total_timesteps = 0

    epsilon = 0.4
    epsilon2 = 0.4
    epsilon2_decay = 0.999
    epsilon_decay = 0.999

    epsilon_min = 0.05
    epsilon2_min = 0.1#0.05

    timestep_list = []
    avg_timestep_list = []

    reward_list = []

    for ep in range(args.total_episodes):
        plt.ion()
        reward_episode = []
        if ep % args.plot_freq ==0:
            

            fig = plt.figure()
            ax = plt.axes(xlim=(0,20),ylim=(-10,10))

            lines, = ax.plot([],[],'o-')
            areas, = ax.fill([],[],'r',alpha=0.1)
            bodyF = ax.scatter([],[],c='r',s=10)
            
            bodyT = ax.scatter([],[],c='g',s=10)


        # state = env.reset()
        episodic_reward = 0
        timestep = 0

        agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
        agentT = SingleIntegrator(np.array([1,0]),dt)

        T_ns_prev = agentT.X
        F_ns_prev = agentF.X

        TX = agentT.X
        FX = agentF.X    
        # state = np.array([F[0,0],F[1,0],F[2,0],T[0,0],T[1,0]])
        state = np.array([FX[2,0]])/np.pi*10.0

        t = 0

        uL = 0.1
        vL = 0.2*np.sin(np.pi*t/5) #  0.1

        # uL = 0.5
        # vL = 3.6*np.sin(np.pi*t) #  0.1 # 1.2

        epsilon = epsilon*epsilon_decay
        if epsilon<epsilon_min:
            epsilon = epsilon_min
        
        # epsilon2 = epsilon2*epsilon2_decay
        # if epsilon2<epsilon2_min:
        #     epsilon2 = epsilon2_min

        epsilon = 0.1
        epsilon2 = 0.4
        epsilon2_decay = 0.999
        epsilon_decay = 0.9

        epsilon_min = 0.05
        epsilon2_min = 0.1#0.05

        obs_d1 = []
        obs_d2 = []
        obs_d3 = []

        true_d1 = []
        true_d2 = []
        true_d3 = []

        pred_d1 = []
        pred_d2 = []
        pred_d3 = []

        h1p = []
        h2p = []
        h3p = []

        debug_value = []

        Xgp = []

        N = 3

        GP_list = build_GP_model(N)

        for st in range(args.max_steps):

            if len(agent_TD3.replay_buffer)==args.batch_size: #st % args.batch_size == 0:
                # print(f"st:{st}, batch_size:{args.batch_size}")
                agent_TD3.replay_buffer = deque(maxlen = args.batch_size)

            epsilon = epsilon*epsilon_decay
            if epsilon<epsilon_min:
                epsilon = epsilon_min
            
            # epsilon2 = epsilon2*epsilon2_decay
            # if epsilon2<epsilon2_min:
            #     epsilon2 = epsilon2_min

            uL = 0.1
            vL = 0.2*np.sin(np.pi*t/5) #  0.1

            # uL = 0.5
            # vL = 3.6*np.sin(np.pi*t) #  0.1 # 1.2

            # uL = 0.5
            # vL = 3.6*np.sin(np.pi*t) #  0.1 # 1.2

            action = agent_TD3.policy(state)

            # # Select action randomly or according to policy
            # # if total_timesteps < args.start_timestep:
            # rand_sample = np.random.random()
            # action_actor = agent_TD3.policy(state)
            # # if (np.random.random()<epsilon):
            # if rand_sample<epsilon:
            #     found_valid = False
            #     max_iter = 100
            #     count_iter = 0
            #     l_prev = 100
            #     # while not found and count_iter<max_iter:
            #     while count_iter<max_iter:
            #         count_iter += 1
            #         u = -3 + 6*random.uniform(0,1)
            #         v = -3 + 6*random.uniform(0,1)
            #         action_temp = [u,v]
            #         # print("actor_temp",action_temp)
            #         valid, l1, l2, l3 = CBF_loss(agentF,agentF,u,v)
            #         l = min(l1,l2,l3)
            #         if valid>=0:
            #             found_valid = True
            #             if l<l_prev:  # take the one nearest the boundary
            #                 l_prev = l
            #                 action = action_temp
            #     # print("PREDICTION",reward_prediction_prev)
            #     # print(f"action: u:{action[0]}, v:{action[1]}")
            #     if not found_valid: # can't help it
            #         print("********************  COULD NOT FIND valid random sample in 100 trials ************************")
            #         # compute_reward(F_ns_sim,T_ns_sim,F_ns_prev,T_ns_prev,type='CBF',debug=True)
            #         # print("prev reward",reward_prev)
            #         action = action_actor #agent.policy(state)
                

            #     # u = -3 + 6*random.uniform(0,1)
            #     # v = -3 + 6*random.uniform(0,1)
            #     # action = [u,v]

            #     #Choose epsilon based on CBF boundary

            # # just choose any safe value
            # elif rand_sample < epsilon2:
            #     valid = False
            #     d_std = 1.0
            #     while not valid:
            #         u = -3 + 6*random.uniform(0,1)
            #         v = -3 + 6*random.uniform(0,1)
            #         action_temp = [u,v]
            #         T_ns_sim = agentT.step_sim(uL,vL)  #agentT.step(0.2,0.5)        
            #         F_ns_sim = agentF.step_sim(np.array([u,v]))
            #         valid, l1, l2, l3 = CBF_loss(agentF,targetF,u,v)
            #     action = action_temp
            # else:
            #     action = action_actor # agent.policy(state)

            #  # still need.. adversarial actions.. explorartion not clear then.. how to do it??
            # # have F_ns_prev, T_ns_prev: oh just do agent .step and do it?? assumes known dynamics then?? 
            # # But do noit know uL, vL?? Now what???

            # # action = agent.policy(state)

            if t==0:
                d1 = 0
                d2 = 0
                d3 = 0
            else:
                d1 = action[0]/10.0
                d2 = action[1]/10.0
                d3 = action[2]/10.0
                # print(f"state:{agentF.X[2][0]}, d1:{d1}, d2:{d2}. d3:{d3}")

            # rand_sample = np.random.random()
            # if rand_sample<epsilon:
            #     success = False
            #     for i in range(100):
            #         d1 = -agent_TD3.upper_bound/20 + 2*agent_TD3.upper_bound/20*random.uniform(0,1)
            #         d2 = -agent_TD3.upper_bound/20 + 2*agent_TD3.upper_bound/20*random.uniform(0,1)
            #         d3 = -agent_TD3.upper_bound/20 + 2*agent_TD3.upper_bound/20*random.uniform(0,1)
            #         f_corrected_temp = agentF.f_corrected
            #         agentF.f_corrected = np.array([d1*np.cos(agentF.X[2][0]),d2*np.sin(agentF.X[2][0]),d3]).reshape(-1,1)
            #         success, u,w = CBF_CLF_QP(agentF,agentT)
            #         if success:
            #             break
            #     if not success:
            #         agentF.f_corrected = f_corrected_temp
            #         print("Could not find safe expoloration point")
            #         d1 = action[0]
            #         d2 = action[1]
            #         d3 = action[2]



            pred_d1.append(d1)
            pred_d2.append(d2)
            pred_d3.append(d3)


            # agentF.f_corrected = np.array([agentF.d1*np.cos(agentF.X[2][0]),agentF.d2*np.sin(agentF.X[2][0]),agentF.d3]).reshape(-1,1)
            # predicted
            agentF.f_corrected = np.array([d1*np.cos(agentF.X[2][0]),d2*np.sin(agentF.X[2][0]),d3]).reshape(-1,1)
            # ignored
            # agentF.f_corrected = np.array([0*np.cos(agentF.X[2][0]),0*np.sin(agentF.X[2][0]),0]).reshape(-1,1)
            agentF.g_corrected = np.array([ [0,0],[0,0],[0,0] ])

            # Recieve state and reward from environment.
            # next_state, reward, done, info = env.step(action)

            # h from previous state:
            h1_prev ,_ ,_ = agentF.CBF1_loss(agentT)
            h2_prev ,_ ,_ = agentF.CBF2_loss(agentT)
            h3_prev ,_ ,_ = agentF.CBF3_loss(agentT)

            FX_prev = agentF.X  

            success, u,w = CBF_CLF_QP(agentF,agentT)
            if not success:
                print("******** ERROR: problem infeasible *************")
            U = np.array([u,w])
            T_ns = agentT.step(uL,vL)  #agentT.step(0.2,0.5)        
            F_ns = agentF.step(U)

            true_d1.append(agentF.d1)#(0.3*np.sin(agentF.X[2])**2) #sin(u)
            true_d2.append(agentF.d2)#(0.3*np.sin(agentF.X[2])**2)
            true_d3.append(agentF.d3)#(0.3)


            FX = agentF.X
            # print("d1 from F",agentF.d1)
            if (np.abs(np.cos(FX_prev[2]))>0.0001):
                d1_obs = ( (FX[0,0]-FX_prev[0,0])/dt - u*np.cos(FX_prev[2,0]) )/np.cos(FX_prev[2,0])
                # print("d1_obs",d1_obs)
                # print(f"d1_obs: {d1_obs}, X:{FX_prev}, X_next:{FX}, gu :{u*np.cos(FX_prev[2])}, f:{ (FX[0]-FX_prev[0])/dt - u*np.cos(FX_prev[2]) } ")
            else:
                # print("0000")
                d1_obs = 0
            if (np.abs(np.sin(FX_prev[2,0]))>0.0001):
                d2_obs = ( (FX[1,0]-FX_prev[1,0])/dt - u*np.sin(FX_prev[2,0]) )/np.sin(FX_prev[2,0])
            else:
                d2_obs = 0
            d3_obs =  wrap_angle(FX[2,0]-FX_prev[2,0])/dt - w
            # print("d1_obs",d1_obs)

            obs_d1.append(d1_obs)
            obs_d2.append(d2_obs)
            obs_d3.append(d3_obs)

            debug_value.append(FX_prev[2])   

            # maximize reward
            reward = compute_reward_model_error( ( d1*10.0 - d1_obs*10.0 ), ( d2*10.0 - d2_obs*10.0 ), ( d3*10.0 - d3_obs*10.0 ) )
            # print("reward: ",reward)
            h1_new , h1A ,_h1B = agentF.CBF1_loss(agentT)
            h1p.append(h1_new)
            h2_new ,h2A , h2B = agentF.CBF2_loss(agentT)
            h2p.append(h2_new)
            h3_new , h3A , h3B = agentF.CBF3_loss(agentT)
            h3p.append(h3_new)

            # h1_dot_true = (h1_new - h1_prev)/dt
            # h2_dot_true = (h2_new - h2_prev)/dt
            # h3_dot_true = (h3_new - h3_prev)/dt

            # U = np.array([u,w]).reshape(-1,1)
            # h1_dot_estimated = h1A @ U + _h1B
            # h2_dot_estimated = h2A @ U + _h2B
            # h3_dot_estimated = h3A @ U + _h3B  

            # reward = compute_reward_V_B_errors( (h1_dot_estimated - h1_dot_true), (h2_dot_estimated - h2_dot_true), (h3_dot_estimated - h3_dot_true) )
    
            reward_episode.append(reward)

            reward_prev = reward
            # next_state = np.array([F[0,0],F[1,0],F[2,0],T[0,0],T[1,0]])
            next_state = np.array([agentF.X[2,0]])/np.pi*10  # state from -1 to 1

            if ep % args.plot_freq ==0:
                lines, areas, bodyF = agentF.render(lines,areas,bodyF)
                bodyT = agentT.render(bodyT)
                
                fig.canvas.draw()
                fig.canvas.flush_events()


            # if reward<0:
            if h1_new<0 or h2_new<0 or h3_new<0:
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
            is_new_data = agent_TD3.filter_data(state, action, reward, next_state, done)
            if is_new_data:
                agent_TD3.add_to_replay_memory(state, action, reward, next_state, done, d1_obs*10.0, d2_obs*10.0, d3_obs*10.0) 

            agent_TD3.replay_buffer.append( ( state, action, reward, next_state, done, d1_obs*10.0, d2_obs*10.0, d3_obs*10.0 ) )
            # print("length",len(agent_TD3.replay_buffer))
            # End this episode when `done` is True
            if done:
                break
            state = next_state
            # print("new state",state)

            timestep += 1  
            total_timesteps += 1

            T_ns_prev = T_ns
            F_ns_prev = F_ns

            t += dt

            if len(agent_TD3.replay_buffer)==args.batch_size:
                print("***************** UPDATING WEIGHTS *****************")
                agent_TD3.learn_and_update_weights_by_passed_replay()
                

        if ep % args.plot_freq ==0:
            plt.close()

        reward_list.append(reward_episode)

        ep_reward_list.append(episodic_reward)
        # Mean of last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)

        timestep_list.append(timestep)

        avg_timestep = np.mean(timestep_list[-100:])
        avg_timestep_list.append(avg_timestep)

        # if avg_reward > 290:
        # if (ep % 1 == 0 and ep>1):# or episodic_reward>70 :
        if episodic_reward>1000:
            # print("here 2")
            #test_reward = eval_policy(agent, seed=88, eval_episodes=1)
            #if test_reward > 30000:
                # print("here 3")
                #final_test_reward = eval_policy(agent, seed=88, eval_episodes=10)
                #if final_test_reward > 300:
                    print("===========================")
                    print('Task Solved')
                    print("===========================")
                    #save weights
                    torch.save(agent.actor.state_dict(), 'actor_CBF.pth')
                    break     
            # if episodic_reward>100:   
            #     torch.save(agent.actor.state_dict(), 'actor_CBF.pth')    
        s = (int)(time.time() - time_start)
        # agent_TD3.learn_and_update_weights_by_replay(timestep,episodic_reward)
        # print("1",total_timesteps)
        # print("2",timestep)
        # print("3",episodic_reward)
        # print("4",avg_reward)
        # print("5",avg_timestep)
        # print("6",epsilon)
        print('Ep. {}, Timestep {},  Ep.Timesteps {}, Buffer size {}, Episode Reward: {:.2f}, Moving Avg.Reward: {:.2f}, Fail Time: {:.2f} , epsilon: {:.2f}, Time: {}:{}:{}'
                .format(ep, total_timesteps, timestep, len(agent_TD3.replay_memory_buffer),
                      episodic_reward, avg_reward, avg_timestep, epsilon, s//3600, s%3600//60, s%60)) 

        plt.figure()
        plt.ioff()
        plt.subplot(3,1,1)
        plt.plot(obs_d1,'g*')
        plt.plot(true_d1,'r')
        plt.plot(pred_d1,'c')

        plt.subplot(3,1,2)
        plt.plot(obs_d2,'g*')
        plt.plot(true_d2,'r')
        plt.plot(pred_d2,'c')

        plt.subplot(3,1,3)
        plt.plot(obs_d3,'g*')
        plt.plot(true_d3,'r')
        plt.plot(pred_d3,'c')
        plt.show()
        # plt.pause(1.0)
        # plt.close()
        # print("Continue?")
        # input_key = input()
        # print(input_key)
        # if input_key == 's':
        #     break
    # Plotting graph
    # Episodes versus Avg. Rewards

    


    # plt.ioff()
    # print("DONE!!")
    # plt.figure()

    # plt.plot(ep_reward_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Epsiodic Reward")
    # plt.show()

    # print("DONE!!")
    # plt.figure()

    # plt.plot(timestep_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Time to Failure")
    # plt.show()

    # print("DONE!!")

    # plt.figure()
    # for reward_episode in reward_list:
    #     plt.plot(reward_episode)
    # plt.show()

    # print("reward list",reward_list)

    # f = open('data_new.dat','wb')

    # for reward_episode in reward_list:
    #     np.savetxt(f,np.asarray(reward_episode))
    #     f.write('\n \n \n')  ## ERROR
    # f.close()


    env.close()

import argparse
parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="BipedalWalkerHardcore-v3")
parser.add_argument('--rl-name', default="td3")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',help='target smoothing coefficient(τ)')
parser.add_argument('--lr-actor', type=float, default=0.003, metavar='G',help='learning rate of actor')  #0.003
parser.add_argument('--lr-critic', type=float, default=0.03, metavar='G',help='learning rate of critic') #0.003
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--seed', type=int, default=123456, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='batch size (default: 256)') #100
parser.add_argument('--buffer-capacity', type=int, default=1000000, metavar='N', help='buffer_capacity')
parser.add_argument('--max-steps', type=int, default=100, metavar='N',help='maximum number of steps of each episode') #10000
parser.add_argument('--total-episodes', type=int, default=6, metavar='N',help='total training episodes') #1000
parser.add_argument('--policy-freq', type=int, default=2, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
args = parser.parse_args("")

train(args)