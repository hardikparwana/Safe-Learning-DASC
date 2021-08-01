import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import os
import time
from numpy.core.numeric import _moveaxis_dispatcher

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import cvxpy as cp
from torch.cuda import max_memory_cached

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *

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



class Actor:
    def __init__(self,alpha=0.1,k=0.1,umax=max_action,umin=min_action) :
        self.alpha = 0.1 #1.0 does not work
        self.k = 0.1
        self.max_action = max_action
        self.min_action = min_action
        self.

    def policy(agent,target):
        u, w = agent.nominal_controller(target)
        U = np.array([u,w]).reshape(-1,1)

        h1,dh1_dxA,dh1_dxB = agent.CBF1_loss(target)
        h2,dh2_dxA,dh2_dxB = agent.CBF2_loss(target)
        h3,dh3_dxA,dh3_dxB = agent.CBF3_loss(target)

        V,dV_dxA,dV_dxB = agent.CLF_loss(target)

        x = cp.Variable((2,1))
        objective = cp.Minimize(cp.sum_squares(x-U))
        
        # CLF constraint
        const = [dV_dxA @ agent.xdot(x) + dV_dxB @ target.xdot(target.U) <= -k*V]

        # CBF constraints
        const += [dh1_dxA @ agent.xdot(x) + dh1_dxB @ target.xdot(target.U) >= -alpha*h1]
        const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha*h2]
        const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha*h3]

        problem = cp.Problem(objective,const)
        result = problem.solve()

        if problem.status == 'optimal':
            return True, np.array([x.value[0,0], x.value[1,0]]).reshape(-1,1)
        else:
            print("SBF QP not solvable")
            return False, np.array([0,0]).reshape(-1,1)

class policy_learning_agent:
    def __init__(self,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4, #actor learning rate
        batch_size = 100,
        buffer_capacity = 100000,
        alpha = 0.1,
        k = 0.1,
        max_action,
        min_action):

            # actor
            self.actor = Actor(alpha=0.1,k=0.1,umax=max_action,umin=min_action)
            self.actor_alpha_left = Actor(alpha=self.actor.alpha-0.05,k=self.actor.k,umax=max_action,umin=min_action)
            self.actor_alpha_right = Actor(alpha=self.actor.alpha+0.05,k=self.actor.k,umax=max_action,umin=min_action)
            self.actor_k_left = Actor(alpha=self.actor.alpha,k=self.actor.k+0.05,umax=max_action,umin=min_action)
            self.actor_k_left = Actor(alpha=self.actor.alpha,k=self.actor.k-0.05,umax=max_action,umin=min_action)
            
            # Replay memory
            self.replay_memory_buffer = deque(maxlen = buffer_capacity)
            self.replay_memory_buffer1 = deque(maxlen = buffer_capacity)
            self.replay_memory_buffer2 = deque(maxlen = buffer_capacity)
            self.replay_memory_buffer3 = deque(maxlen = buffer_capacity)
            self.replay_memory_buffer4 = deque(maxlen = buffer_capacity)

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        #add samples to replay memory
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def get_random_sample_from_replay_mem(self,timesteps=None):
        #random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def learn_and_update_weights_by_replay(self,training_iterations):
        print("hello")

    def policy(follower,target):
        U = self.actor.policy(follower,target)
        U_alpha_left = self.actor_alpha_left.policy(follower,target)
        U_alpha_right = self.actor_alpha_right.policy(follower,target)
        U_k_left = self.actor_k_left.policy(follower,target)
        U_k_right = self.actor_k_right.policy(follower,target)

        return U, U_alpha_left, U_alpha_right, U_k_left, U_k_right

def train(args):

    # Exploration Parameters
    epsilon = 0.4
    epsilon_decay = 0.999
    epsilon_min = 0.05

    epsilon2 = 0.4
    epsilon2_decay = 0.999
    epsilon2_min = 0.1#0.05

    timestep_list = []
    avg_timestep_list = []

    reward_list = []

    dt = 0.1

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

        max_action = np.array([3,3]).reshape(-1,1)
        min_action = np.array([-3,-3]).reshape(-1,1)
        
        agent = policy_learning_agent(gamma = args.gamma, #discount factor
            lr_actor = args.lr_actor, #actor learning rate
            batch_size = args.batch_size,
            buffer_capacity = args.buffer_capacity,
            alpha = 0.1,
            k = 0.1,
            max_action=max_action,
            min_action=min_action)
        
        agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
        agentT = SingleIntegrator(np.array([1,0]),dt)

        TX_prev = agentT.X
        FX_prev = agentF.X

        TX = agentT.X
        FX = agentF.X    
        state = np.array([FX[0,0],FX[1,0],FX[2,0],TX[0,0],TX[1,0]])

        t = 0

        for st in range(args.max_steps):  # for each step, move 10 times??

            # re-initialize everything if horizon reached
            if st % args.horizon == 0:
                agentF_alpha_left = Unicycle2D(agentF.X,dt,3,FoV,max_D,min_D)
                agentF_alpha_right = Unicycle2D(agentF.X,dt,3,FoV,max_D,min_D)
                agentF_k_left = Unicycle2D(agentF.X,dt,3,FoV,max_D,min_D)
                agentF_k_right = Unicycle2D(agentF.X,dt,3,FoV,max_D,min_D)

                agent.replay_memory_buffer = deque(maxlen = buffer_capacity)
                agent.replay_memory_buffer1 = deque(maxlen = buffer_capacity)
                agent.replay_memory_buffer2 = deque(maxlen = buffer_capacity)
                agent.replay_memory_buffer3 = deque(maxlen = buffer_capacity)
                agent.replay_memory_buffer4 = deque(maxlen = buffer_capacity)

            epsilon = epsilon*epsilon_decay
            if epsilon<epsilon_min:
                epsilon = epsilon_min
            
            epsilon2 = epsilon2*epsilon2_decay
            if epsilon2<epsilon2_min:
                epsilon2 = epsilon2_min

            uL = 0.1
            vL = 0.2*np.sin(np.pi*t/5) #  0.1

            # Exploration vs Exploitation
            rand_sample = np.random.random()
            if rand_sample<epsilon:
                found_valid =  False
                ...
            else:
                action, action1, action2, action3, action4 = agent.policy(state)


            # Propagate state
            U_L = np.array([uL,vL]).reshape(-1,1)
            FX = agentF.step(action.reshape(-1,1))
            TX = agentT.step(action.reshape(-1,1))

            # other parameters:
            FX1 = agentF_alpha_left.step(action1.reshape(-1,1))
            FX2 = agentF_alpha_right.step(action2.reshape(-1,1))
            FX3 = agentF_k_left.step(action3.reshape(-1,1))
            FX4 = agentF_k_right.step(action4.reshape(-1,1))

            # Compute reward
            reward = agentF.compute_reward(agentT)
            reward_episode.append(reward)
            reward_prev = reward
            episodic_reward += reward

            ## Other parameters
            reward1 = agentF_alpha_left.compute_reward(agentT)
            reward2 = agentF_alpha_right.compute_reward(agentT)
            reward3 = agentF_k_left.compute_reward(agentT)
            reward4 = agentF_k_right.compute_reward(agentT)

            # add to buffer
            next_state = np.array([FX[0,0],FX[1,0],FX[2,0],TX[0,0],TX[1,0]])

            next_state1 = np.array([FX1[0,0],FX1[1,0],FX1[2,0],TX[0,0],TX[1,0]])
            next_state2 = np.array([FX2[0,0],FX2[1,0],FX2[2,0],TX[0,0],TX[1,0]])
            next_state3 = np.array([FX3[0,0],FX3[1,0],FX3[2,0],TX[0,0],TX[1,0]])
            next_state4 = np.array([FX4[0,0],FX4[1,0],FX4[2,0],TX[0,0],TX[1,0]])
            

            # animation plot
            if ep % args.plot_freq ==0:
                lines, areas, bodyF = agentF.render(lines,areas,bodyF)
                bodyT = agentT.render(bodyT)
                
                fig.canvas.draw()
                fig.canvas.flush_events()

            if reward<0:
                done = True
            else:
                done = False

            agent.add_to_replay_memory(state, action, reward, next_state, done)
            agent.add_to_replay_memory1(state1, action1, reward1, next_state1, done1)
            agent.add_to_replay_memory2(state2, action2, reward2, next_state2, done2)
            agent.add_to_replay_memory3(state3, action3, reward3, next_state3, done3)
            agent.add_to_replay_memory4(state4, action4, reward4, next_state4, done4)

            if done:
                break

            # Update loop variables
            state = next_state
            state1 = next_state1
            state2 = next_state2
            state3 = next_state3
            state4 = next_state4
            timestep += 1     
            total_timesteps += 1
            TX_prev = TX
            FX_prev = FX
            t += dt

        print("became unsafe")
        reward_list.append(reward_episode)


            
                



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