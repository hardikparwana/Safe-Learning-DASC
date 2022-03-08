# SIngle Integrator example

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import matplotlib.patches as mpatches

import os
import time
from numpy.core.numeric import _moveaxis_dispatcher

import cvxpy as cp
import torch

from cvxpylayers.torch import CvxpyLayer

from robot_models.Unicycle2D import *
from robot_models.SingleIntegrator import *

from matplotlib.animation import FFMpegWriter

FoV = 60*np.pi/180
max_D = 3
min_D = 0.3

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def wrap_angle(angle):
    if angle>np.pi:
        angle = angle - 2*np.pi
    if angle<-np.pi:
        angle = angle + 2*np.pi
    return angle


class Actor:
    def __init__(self,alpha=0.1,k=0.1,umax=np.array([3,3]),umin=np.array([-3,-3])) :
        self.alpha_nominal = alpha #1.0 does not work
        self.alpha1_nominal = alpha
        self.alpha2_nominal = alpha
        self.alpha3_nominal = alpha
        self.k_nominal = k
        self.max_action = umax
        self.min_action = umin

    def policy(self,agent,target):
       
        u, w = agent.nominal_controller(target)
        U = np.array([0,0]).reshape(-1,1)

        h1,dh1_dxA,dh1_dxB = agent.CBF1_loss(target)
        h2,dh2_dxA,dh2_dxB = agent.CBF2_loss(target)
        h3,dh3_dxA,dh3_dxB = agent.CBF3_loss(target)

        V,dV_dxA,dV_dxB = agent.CLF_loss(target)

        x = cp.Variable((2,1))
        delta = cp.Variable(1)
        objective = cp.Minimize(cp.sum_squares(x-U) + 100*delta)

        alpha1 = cp.Parameter(value=self.alpha1_nominal)
        alpha2 = cp.Parameter(value=self.alpha2_nominal)
        alpha3 = cp.Parameter(value=self.alpha3_nominal)
        k = cp.Parameter(value=self.k_nominal)
        
        # CLF constraint
        const = [dV_dxA @ agent.xdot(x) + dV_dxB @ target.xdot(target.U) <= -k*V + delta]
        # print("Lyapuniov function V",V)
        const += [delta>=0]

        epsilon = 0.9
        
        const += [dh1_dxA @ agent.xdot(x) + dh1_dxB @ target.xdot(target.U) >= -alpha1*h1 + 0.001]#np.linalg.norm(dh1_dxA @ (agent.g+ agent.g_corrected))/epsilon]
        const += [dh2_dxA @ agent.xdot(x) + dh2_dxB @ target.xdot(target.U) >= -alpha2*h2 + 0.001]#np.linalg.norm(dh2_dxA @ (agent.g+ agent.g_corrected))/epsilon]
        const += [dh3_dxA @ agent.xdot(x) + dh3_dxB @ target.xdot(target.U) >= -alpha3*h3 + 0.001]#np.linalg.norm(dh3_dxA @ (agent.g+ agent.g_corrected))/epsilon]
        
        const += [cp.abs(x[0,0])<=self.max_action[0]  ]
        const += [cp.abs(x[1,0])<=self.max_action[1]  ]
        
        
        problem = cp.Problem(objective,const)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[alpha1, alpha2, alpha3, k], variables=[x])
        alpha1_tch = torch.tensor(alpha1.value, requires_grad=True, dtype=torch.float)
        alpha2_tch = torch.tensor(alpha2.value, requires_grad=True, dtype=torch.float)
        alpha3_tch = torch.tensor(alpha3.value, requires_grad=True, dtype=torch.float)
        k_tch = torch.tensor(k.value, requires_grad=True, dtype=torch.float)


        solver_args = {
            'verbose': False,
            'max_iters': 1000000
        }

        # Solve the QP
        try:
            solution, = cvxpylayer(alpha1_tch, alpha2_tch, alpha3_tch, k_tch, solver_args=solver_args)
        except:
            print("SBF QP not solvable")
            return False, np.array([0,0]).reshape(-1,1), 0, 0

        ### Gradient computation
        # solution.sum().backward()

        e1 = torch.tensor(np.array([1.0,0]), dtype=torch.float)
        x_value = solution.detach().numpy()
        u1 = torch.matmul(e1,solution)
        u1.backward()
        
        u1_alpha1 = np.copy(alpha1_tch.grad.numpy().reshape(1,-1))
        u1_alpha2 = np.copy(alpha2_tch.grad.numpy().reshape(1,-1))
        u1_alpha3 = np.copy(alpha3_tch.grad.numpy().reshape(1,-1))
        u1_k = np.copy(k_tch.grad.numpy().reshape(1,-1))
        u1_grad = np.append(  u1_alpha1, u1_alpha2, axis=1  )
        u1_grad = np.append( u1_grad, u1_alpha3, axis = 1 )
        u1_grad = np.append( u1_grad, u1_k , axis = 1 )
        

        e2 = torch.tensor(np.array([0.0,1.0]), dtype=torch.float)        
        u2 = torch.matmul(e2,solution)
        u2.backward()

        u2_alpha1 = np.copy(alpha1_tch.grad.numpy().reshape(1,-1)) - u1_alpha1
        u2_alpha2 = np.copy(alpha2_tch.grad.numpy().reshape(1,-1)) - u1_alpha2
        u2_alpha3 = np.copy(alpha3_tch.grad.numpy().reshape(1,-1)) - u1_alpha3
        u2_k = np.copy(k_tch.grad.numpy().reshape(1,-1)) - u1_k
    
        u2_grad = np.append(  u2_alpha1, u2_alpha2, axis=1  )
        u2_grad = np.append( u2_grad, u2_alpha3, axis = 1 )
        u2_grad = np.append( u2_grad, u2_k , axis = 1 )

        u_grad = np.append(u1_grad, u2_grad, axis=0)
        
        return True, solution.detach().numpy().reshape(-1,1), u_grad, delta.value

class policy_learning_agent:
    def __init__(self, 
        max_action,
        min_action,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4, #actor learning rate
        batch_size = 100,
        buffer_capacity = 100000,
        alpha = 2.0,
        k = 0.1,
        beta = 0.05
        ):
            #buffer
            self.buffer_capacity = buffer_capacity

            #learning rate
            self.beta = beta

            # actor
            self.actor = Actor(alpha=alpha,k=k,umax=max_action,umin=min_action)
            
            # Replay memory
            self.replay_memory_buffer = deque(maxlen = buffer_capacity)

    def add_to_replay_memory(self, state, action, state_action_grad, reward, reward_grad, next_state, done, param_grad):
        #add samples to replay memory
        self.replay_memory_buffer.append((state, action, state_action_grad, reward, reward_grad, next_state, done, param_grad))

    def get_random_sample_from_replay_mem(self,timesteps=None):
        #random samples from replay memory
        random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
        return random_sample

    def learn_and_update_weights_by_multiple_shooting(self):
        # Update alpha and k here

        #replay buffer order: state, action, state_action_grad, reward, reward_grad, next_state, done, param_grad

        # assert len(self.replay_memory_buffer)==self.horizon

        policy_gradient = []
        for index, value in enumerate(self.replay_memory_buffer): #index from 0 to horizon -1
            state_action_grad = value[2]
            reward_grad = value[4]
            qp_param_grad = value[7]
            policy_gradient.append( reward_grad @ state_action_grad @ qp_param_grad  )
            

        policy_gradient = np.asarray(policy_gradient)
        policy_gradient = np.sum(policy_gradient,axis = 0)

        self.actor.alpha1_nominal = self.actor.alpha1_nominal + self.beta*policy_gradient[0]
        self.actor.alpha2_nominal = self.actor.alpha2_nominal + self.beta*policy_gradient[1]
        self.actor.alpha3_nominal = self.actor.alpha3_nominal + self.beta*policy_gradient[2]
        self.actor.k_nominal = self.actor.k_nominal + self.beta*policy_gradient[3]

        # clipping > 0
        if self.actor.alpha1_nominal < 0:
            self.actor.alpha1_nominal = 0
        if self.actor.alpha2_nominal < 0:
            self.actor.alpha2_nominal = 0
        if self.actor.alpha3_nominal < 0:
            self.actor.alpha3_nominal = 0
        if self.actor.k_nominal < 0:
            self.actor.k_nominal = 0
        
    def policy(self,follower,target):
        solved, U, param_grad, delta = self.actor.policy(follower,target)
        return solved, U, param_grad, delta

def train(args):

    dt = 0.01

    for ep in range(args.total_episodes):
        plt.ion()
        reward_episode = []
        if ep % args.plot_freq ==0:
            fig = plt.figure()
            ax = plt.axes(xlim=(0,10),ylim=(-5,5))
            lines, = ax.plot([],[],'o-')
            poly = mpatches.Polygon([(0,0.2)], closed=True, color='r',alpha=0.1, linewidth=0) #[] is Nx2
            fov_arc = ax.add_patch(poly)
            areas, = ax.fill([],[],'r',alpha=0.1)
            bodyF = ax.scatter([],[],c='r',s=10)            
            bodyT = ax.scatter([],[],c='g',s=10)
            des_point = ax.scatter([],[],s=10, facecolors='none', edgecolors='g')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect(1)
        if args.movie==True:
            frames = [] # for storing the generated images
    
        # state = env.reset()
        episodic_reward = 0
        timestep = 0

        max_action = np.array([40.0,40.0]).reshape(-1,1)
        min_action = np.array([-1.5,-1.5]).reshape(-1,1)
        
        agent = policy_learning_agent(gamma = args.gamma, #discount factor
            lr_actor = args.lr_actor, #actor learning rate
            batch_size = args.batch_size,
            buffer_capacity = args.buffer_capacity,
            alpha = args.alpha,#2.0,
            k = args.k, #0.1,
            beta = args.lr_actor,
            max_action=max_action,
            min_action=min_action)
        
        agentF = Unicycle2D(np.array([0,0.2,0]),dt,3,FoV,max_D,min_D)
        agentT = SingleIntegrator(np.array([1,0]),dt,ax,0)

        TX_prev = agentT.X
        FX_prev = agentF.X

        TX = agentT.X
        FX = agentF.X    
        state = np.array([FX[0,0],FX[1,0],FX[2,0],TX[0,0],TX[1,0]])

        t = 0
        t_plot = []

        reward_horizon = 0
        reward_moving_avg = []

        alphas = []
        alpha1s = []
        alpha2s = []
        alpha3s = []
        ks = []
        deltas = []

        h1s = []
        h2s = []
        h3s = []

        TXs = []

        actions = []

        metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        first_update =  False
        # with writer.saving(fig, args.movie_name, 100): 
        if 1:
            for st in range(args.max_steps):  # for each step, move 10 times??

                # re-initialize everything if horizon reached
                if st % args.horizon == 0:
                    reward_horizon = 0
                    # agent.replay_memory_buffer = deque(maxlen = args.buffer_capacity)


                uL = 1.0
                vL = 12*np.sin(np.pi*t*4) #  0.1 # 1.2

                # Exploration vs Exploitation
                rand_sample = np.random.random()
                if 0:#and_sample<epsilon:
                    found_valid =  False
                    action = np.array([0,0]).reshape(-1,1)
                    param_grad = np.array([[0,0],[0,0]])
                    solved = False
                else:
                    solved, action, param_grad, delta = agent.policy(agentF,agentT)
                    if solved==False:
                        print("************ ERROR: problem infeasible **************")
                        break;

              
                # Propagate state
                state_action_grad = agentF.g
                U_L = np.array([uL,vL]).reshape(-1,1)
                FX = agentF.step(action.reshape(-1,1))
                TX = agentT.step(uL,vL)

                # Compute reward
                reward, reward_grad, h1, h2, h3 = agentF.compute_reward(agentT)
                reward_horizon += reward
                reward_episode.append(reward)
                reward_prev = reward
                episodic_reward += reward

                # add to buffer
                next_state = np.array([FX[0,0],FX[1,0],FX[2,0],TX[0,0],TX[1,0]])
                

                # animation plot
                if ep % args.plot_freq ==0:
                    lines, areas, bodyF, poly, des_point = agentF.render(lines,areas,bodyF, poly, des_point)
                    bodyT = agentT.render(bodyT)
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # if args.movie:
                    #     writer.grab_frame()

                if reward<0:
                    done = True
                else:
                    done = False

                agent.add_to_replay_memory(state, action, state_action_grad, reward, reward_grad, next_state, done, param_grad)

                # Update loop variables
                state = next_state

                timestep += 1     
                TX_prev = TX
                FX_prev = FX
                t += dt
                
                    
                alphas.append(agent.actor.alpha_nominal)
                alpha1s.append(agent.actor.alpha1_nominal)
                alpha2s.append(agent.actor.alpha2_nominal)
                alpha3s.append(agent.actor.alpha3_nominal)
                ks.append(agent.actor.k_nominal)
                t_plot.append(t)
                deltas.append(delta)
                h1s.append(h1)
                h2s.append(h2)
                h3s.append(h3)
                TXs.append(TX)
                actions.append(action)

                if done:
                    print("Became Unsafe")
                    break
                
                # if (st+1) % args.horizon == 0 and st>=(args.buffer_capacity-1) and args.train==True and st>=(args.max_horizon-1):
                if st>=(args.buffer_capacity-1) and args.train==True and st>=(args.max_horizon-1):
                    if not first_update:
                        print(f"st:{st}, term1: {(st+1) % args.horizon}, horizon:{args.horizon}, buffer:{args.buffer_capacity}, train:{args.train}, max horixon:{args.max_horizon}")
                        first_update = True
                    # print(len(agent.replay_memory_buffer))
                    # print("update at: ",st)
                    reward_moving_avg.append(reward_horizon)
                    agent.learn_and_update_weights_by_multiple_shooting()

        return reward_episode, reward_moving_avg, alphas, ks, t_plot, deltas, h1s, h2s, h3s, TXs, actions, alpha1s, alpha2s, alpha3s

        


            
                



import argparse
parser = argparse.ArgumentParser(description='td3')
parser.add_argument('--env-name', default="BipedalWalkerHardcore-v3")
parser.add_argument('--rl-name', default="td3")
parser.add_argument('--gamma', type=float, default=0.99,metavar='G',help='discounted factor')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',help='target smoothing coefficient(τ)')
parser.add_argument('--lr_actor', type=float, default=0.03, metavar='G',help='learning rate of actor')  #0.003
parser.add_argument('--lr-critic', type=float, default=0.03, metavar='G',help='learning rate of critic') #0.003
parser.add_argument('--plot_freq', type=float, default=1, metavar='G',help='plotting frequency')
parser.add_argument('--seed', type=int, default=123456, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='batch size (default: 256)') #100
parser.add_argument('--buffer_capacity', type=int, default=30, metavar='N', help='buffer_capacity') #10
parser.add_argument('--max-steps', type=int, default=200, metavar='N',help='maximum number of steps of each episode') #70
parser.add_argument('--total-episodes', type=int, default=1, metavar='N',help='total training episodes') #1000
parser.add_argument('--policy-freq', type=int, default=500, metavar='N',help='update frequency of target network ')
parser.add_argument('--start-timestep', type=int, default=10000, metavar='N',help='number of steps using random policy')
parser.add_argument('--horizon', type=int, default=10, metavar='N',help='RL time horizon') #3
parser.add_argument('--max_horizon', type=int, default=30, metavar='N',help='RL time horizon') #3
parser.add_argument('--alpha', type=float, default=0.15, metavar='G',help='CBF parameter')  #0.003
parser.add_argument('--k', type=float, default=0.1, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--train', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie', type=float, default=True, metavar='G',help='CLF parameter')  #0.003
parser.add_argument('--movie_name', default="test_temp.mp4")
args = parser.parse_args("")

Alphas = [0.15]#, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]#[0.0] #0.15 #0.115
Ks = [0.1] #0.1 #2.0
Trains = [True, False]
Betas = [0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
movie_names = ['Adaptive.mp4','Non-Adaptive.mp4']

horizons = [10, 1, 2, 3, 4, 5, 10, 20, 30]
horizons = [10, 1, 5, 10, 15, 20, 25, 30]

reward_episodes = []
reward_horizons = []

# plt.figure()
figure1, axis1 = plt.subplots(2, 2)

# plt.figure()
figure2, axis2 = plt.subplots(1, 1)

# plt.figure()
figure3, axis3 = plt.subplots(1, 1)

figure4, axis4 = plt.subplots(2, 1)

figure5, axis5 = plt.subplots(2, 1)

colors = ['k','maroon','red','salmon','yellow','yellowgreen','darkgreen','cornflowerblue','blue']
# colors = ['r','g','b','k','c','m']
colors2 = colors.copy()
colors2.reverse()

index = 0
data = []
for Alpha in Alphas:
    for K in Ks:
        for ibeta, Beta in enumerate(Betas):
            
            args.alpha = Alpha
            args.k = K
            args.lr_actor = Beta
            args.train = True
            args.horizon = horizons[ibeta]
            # args.movie_name = movie_names[index]

            if index==0:
                name = 'horizon = 0'
                args.buffer_capacity = 1
                args.train = False
            else:
                name = 'horizon = ' + str(horizons[ibeta])
                args.buffer_capacity = args.horizon
            # name = 'test'
            
            episode_reward, moving_reward, alphas, ks, t_plot, deltas, h1s, h2s, h3s, TXs, actions, alpha1s, alpha2s, alpha3s =  train(args)
            data.append( (episode_reward, moving_reward, alphas, ks, t_plot, deltas, h1s, h2s, h3s, TXs, actions, alpha1s, alpha2s, alpha3s) )
            # Reward Plot
            axis2.plot(t_plot,episode_reward,c = colors[index],label = name)

            # Parameter Plot
            axis1[0,0].plot(t_plot,alpha1s,c = colors[index],label = name)
            axis1[1,0].plot(t_plot,alpha2s,c = colors[index],label = name)
            axis1[0,1].plot(t_plot,alpha3s,c = colors[index],label = name)
            axis1[1,1].plot(t_plot,ks,c = colors[index],label = name)

            # axis2[1].plot(t_plot,deltas,c = 'r',label='slack')

            # if index == 0:
            #     mark = '--'
            # if index == 1:
            #     mark = '.'
            # Barrier Function Plots
            axis3.plot(t_plot,h1s,colors[index])
            # style = colors[index]+'.'
            # print(style)
            axis3.plot(t_plot,h2s,colors[index], linestyle='dashed')
            axis3.plot(t_plot,h3s,colors[index],linestyle='dotted')

            # Target Movement Plot
            axis4[0].plot(t_plot,[x[0] for x in TXs],c = 'r',label='X')
            axis4[1].plot(t_plot,[x[1] for x in TXs],c = 'g',label='Y')

            # Control Input plot
            axis5[0].plot(t_plot,[x[0] for x in actions],c = colors[index],label = name)
            axis5[1].plot(t_plot,[x[1] for x in actions],c = colors[index],label = name)

            index += 1

plt.ioff()

import matplotlib 
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

# Parameter Plot
axis1[0,0].set_title(r"$\alpha_1 $")
axis1[1,0].set_title(r"$\alpha_2 $")
axis1[1,0].set_xlabel('time (s)')
axis1[0,1].set_title(r"$\alpha_3 $")
axis1[0,1].legend()
axis1[1,1].set_title(r"$k$")
axis1[1,1].set_xlabel('time (s)')

# figure1.savefig("unicycle_parameter.png")
# figure1.savefig("unicycle_parameter.eps")

# Reward Plot
# axis2.set_title("Reward with time")#,y=1.0,pad=-14)
axis2.set_xlabel('time (s)')
axis2.set_xlabel('Horizon Reward with Time')
axis2.legend()
# figure2.savefig("unicycle_reward.png")
# figure2.savefig("unicycle_reward.eps")


# Barrier Function Plots
axis3.set_title("Barrier Functions")#,y=1.0,pad=-14)
axis3.set_xlabel('time (s)')
axis3.legend()
# figure3.savefig("unicycle_barrier.png")
figure3.savefig("unicycle_barrier2.eps")

axis4[0].set_title('Target Position')
axis4[0].legend()
axis4[1].legend()
axis4[1].set_xlabel('time step')
# figure4.savefig("unicycle_positions.png")
# figure4.savefig("unicycle_positions.eps")

axis5[0].set_title('Control Inputs')
axis5[0].legend()
axis5[0].set_ylabel('Linear Velocity')
axis5[1].set_ylabel('Angular Velocity')
axis5[1].legend()
axis5[1].set_xlabel('time (s)')
# figure5.savefig("unicycle_inputs.png")
# figure5.savefig("unicycle_inputs.eps")

plt.show()
            



