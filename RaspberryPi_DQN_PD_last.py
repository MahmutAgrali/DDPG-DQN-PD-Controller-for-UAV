from math import sin, cos, pi, sqrt, remainder
from gym import Env
from gym.spaces import Discrete, Box,Dict
import random
import matplotlib.pyplot as plt
import openpyxl
from gym.spaces.space import Space
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Add,ReLU, Flatten,Reshape,Input,concatenate,RNN
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, tf
from rl.memory import SequentialMemory
from tensorflow.keras.models import Model
import os
from datetime import datetime
from numpy.random import seed
import tensorflow as tf

seed_value = 42                   # Set seed number
seed(seed_value)                  # Set seed number
tf.random.set_seed(seed_value)    # Set seed number
os.environ['PYTHONHASHSEED']='0'  # Set seed number
np.random.seed(seed_value)        # Set seed number

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)    # Get directory
    if not os.path.exists(directory):         # Check if directory exist
        os.makedirs(directory)                # Generate directory

class UAVEnv(Env):
    def __init__(self,l_action=0,h_action=10,delta_action=0.1,action_list=None,deltaT=0.1, tot_time=1,control_alg="dqn"):
        if(control_alg not in ["dqn","pd"]):
           raise AssertionError(f"There is no control algorithm such as {control_alg}, you can use only dqn and pd algorithms.")
        # Get parameters
        self.deltaT          = deltaT
        self.tot_time        = tot_time
        self.current_episode = 0
        self.control_alg     = control_alg

        # Initialize env
        self.initializeUAV_const()
        self.initializePD_const()
        self.initialize_sim()
        self.set_desired_values()

        # Get date
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        dt_string = f"results/{dt_string}/"
       
        # Get file name
        self.fig_dir =  dt_string +"fig/"
        self.xlsx_dir = dt_string +"xlsx/"

        # Generate directory
        ensure_dir(dt_string)
        ensure_dir(self.fig_dir)
        ensure_dir(self.xlsx_dir)
        
        if(self.control_alg == "dqn"):
            self.model_dir = dt_string +"model/"
        
            ensure_dir(self.model_dir)

            # Get action space
            if action_list ==  None:
                start = round(l_action / delta_action)
                stop = round(h_action / delta_action)
                self.action_values = np.arange(start, stop)
                self.action_values = self.action_values * delta_action
                self.action_space = Discrete(int(stop - start))  # buuuu
            else:
                self.action_values = np.array(action_list)
                self.action_space = Discrete(len(action_list)) # buuuu
            # Get observation space
            self.observation_space = Box(low=-100,high=100,shape=(4,))#Dict(spaces)
            self.state = self.observation_space.sample()

    def initializePD_const(self):
        # Get PD COEFFICIENTS
        self.Kpp = 30
        self.Kdp = 5
        self.Kpt = 30
        self.Kdt = 5
        self.Kpps = 30
        self.Kdps = 5
        self.Kpz = 40
        self.Kdz = 12

    def initializeUAV_const(self):
        # Get UAV COEFFICIENTS
        self.Ixx = 8.1e-3
        self.Iyy = 8.1e-3
        self.Izz = 14.2e-3
        self.Jtp = 104e-6
        self.b = 54.2e-6
        self.d = 1.1e-6
        self.l = 0.24
        self.m = 1.0
        self.g = 9.81

    def initialize_sim(self):
        if(self.deltaT > 1):
            raise ValueError("deltaT value must be less than 1 or equal to 1, but given value is {}".format(self.deltaT))

        # Initilize parameters
        self.total_time_step = int(self.tot_time/self.deltaT)
        self.time_step = 0

        self.x0 = np.zeros(12)
        self.x = list()

        Zinit = 0
        Phiinit = 0
        Thetainit = 0
        Psiinit = 0
        self.x0[0] = 0  # Xinit
        self.x0[2] = 0  # Yinit
        self.x0[4] = Zinit  # Zinit
        self.x0[6] = Phiinit  # Phiinit
        self.x0[8] = Thetainit  # Thetainit
        self.x0[10] = Psiinit  # Psiinit

        self.integral_error = 0


    def set_desired_values(self):
        # Set desired values
        self.Zd = 0.5
        self.Phid = 0.5
        self.Thetad = 0.5
        self.Psid = 0.5

    def time_step_run(self, action=None):
        if(self.control_alg.__eq__("dqn")):
            # Get action if dqn control is enabled
            action = np.squeeze(action)
            action = self.action_values[action]
        # Check first time step
        if (self.time_step == 0):
            x = self.x0
        else:
            x = self.x[-1]

        # Bounding the angles within the - 2 * pi / 2 * pi range
        if (x[6] > 2 * pi or x[6] < - 2 * pi):
            x[6] = remainder(x[6], 2 * pi)

        if (x[8] > 2 * pi or x[8] < - 2 * pi):
            x[8] = remainder(x[8], 2 * pi)

        if (x[10] > 2 * pi or x[10] < - 2 * pi):
            x[10] = remainder(x[10], 2 * pi)

        # Check PD or DQN Control for Z
        U = list()
        if (self.control_alg.__eq__("dqn")):
            U.append(self.m * (self.g +action) / (cos(x[8]) * cos(x[6])))  # Total Thrust on the body along z - axis

        elif(self.control_alg.__eq__("pd")):
            U.append(self.m * (self.g + self.Kpz * (self.Zd - x[4]) + self.Kdz * (- x[5])) / (cos(x[8]) * cos(x[6])))  # Total Thrust on the body along z - axis

        U.append(self.Ixx * (self.Kpp * (self.Phid - x[6]) + self.Kdp * (- x[7])))  # Roll input
        U.append(self.Iyy * (self.Kpt * (self.Thetad - x[8]) + self.Kdt * (- x[9])))  # Pitch input
        U.append(self.Izz * (self.Kpps * (self.Psid - x[10]) + self.Kdps * ( - x[11])))  # Yawing moment csignal[k,:)=U;
        U = np.real(U)

        # Bounding the controls
        if U[0] > 15.7:
            U[0] = 15.7

        if U[0] < 0:
            U[0] = 0

        for j in range(1, 3):
            if U[j] > 1:
                U[j] = 1

            if U[j] < -1:
                U[j] = -1
        omegasqr = list()
        # Calculation of rotors angular velocities
        omegasqr.append((1 / (4 * self.b)) * U[0] + (1 / (2 * self.b * self.l)) * U[2] - (1 / (4 * self.d)) * U[3])
        omegasqr.append((1 / (4 * self.b)) * U[0] - (1 / (2 * self.b * self.l)) * U[1] + (1 / (4 * self.d)) * U[3])
        omegasqr.append((1 / (4 * self.b)) * U[0] - (1 / (2 * self.b * self.l)) * U[2] - (1 / (4 * self.d)) * U[3])
        omegasqr.append((1 / (4 * self.b)) * U[0] + (1 / (2 * self.b * self.l)) * U[1] + (1 / (4 * self.d)) * U[3])
        omegasqr = np.abs(omegasqr)

        omega = list()
        omega.append(sqrt(omegasqr[0]))
        omega.append(sqrt(omegasqr[1]))
        omega.append(sqrt(omegasqr[2]))
        omega.append(sqrt(omegasqr[3]))

        Omega = (-omega[0] + omega[1] - omega[2] + omega[3])

        fc = list()
        # Evaluation of the State space wrt H - frame
        fc.append(x[1])  # Xdot
        fc.append((sin(x[10]) * sin(x[6]) + cos(x[10]) * sin(x[8]) * cos(x[6])) * (U[0] / self.m))  # Xdotdot
        fc.append(x[3])  # Ydotx
        fc.append((-cos(x[10]) * sin(x[6]) + sin(x[10]) * sin(x[8]) * cos(x[6])) * (U[0] / self.m))  # Ydotdot
        fc.append(x[5])  # Zdot
        fc.append(- self.g + (cos(x[8]) * cos(x[6])) * (U[0] / self.m))  # Zdotdot
        fc.append(x[7])  # phydot
        fc.append(((self.Iyy - self.Izz) / self.Ixx) * x[9] * x[11] - (self.Jtp / self.Ixx) * x[9] * Omega + (U[1] / self.Ixx))  # pdot = phydotdot
        fc.append(x[9])  # thetadot
        fc.append(((self.Izz - self.Ixx) / self.Iyy) * x[7] * x[11] + (self.Jtp / self.Iyy) * x[7] * Omega + (U[2] / self.Iyy))  # qdot = thetadotdot
        fc.append(x[11])  # psidot
        fc.append(((self.Ixx - self.Iyy) / self.Izz) * x[7] * x[9] + (U[3] / self.Izz))  # rdot = psidotdot

        x = x + np.dot(self.deltaT, fc)

        self.x.append(x)

        self.time_step += 1

        if (self.control_alg.__eq__("dqn")):
            if (x[4] < 0):
                x[4] = 0
            # Get error and etc.
            error = self.Zd - x[4]
            self.integral_error += error*self.deltaT
            derivative_error = error/self.deltaT

            self.state = np.array([self.integral_error, derivative_error, error, x[4],action])

            reward = -10 * error ** 2
            if (self.time_step == self.total_time_step):
                done = True
                self.current_episode += 1
                # Save plot and episode
                self.update_plot(self.x, self.current_episode)
                self.save_episode(self.current_episode)

            else:
                done = False

            return self.state, reward, done, {}

    def step(self, action):
        return self.time_step_run(action)

    def initialize_plot(self, verbose=False):
        if verbose:
            plt.ion()
        self.generate_plot()
        
    def generate_plot(self):
        # Generate plot
        ini_arr = np.arange(self.total_time_step)
        ini_arr = np.multiply(ini_arr, self.deltaT)
        
        if(self.control_alg.__eq__("dqn")):
            arrZ = ini_arr
            arrPhi = ini_arr
            arrTheta = ini_arr
            arrPsi = ini_arr
        elif(self.control_alg.__eq__("pd")):
            x = np.array(self.x)
            arrZ = x[:,4]
            arrPhi = x[:,6]
            arrTheta = x[:,8]
            arrPsi = x[:,10]       
            
        self.fig = plt.figure()     
        ax = self.fig.add_subplot(111)

        self.lineZ, = ax.plot(ini_arr,arrZ, 'r-')
        self.linePhi, = ax.plot(ini_arr,arrPhi, 'g-')
        self.lineTheta, = ax.plot(ini_arr,arrTheta, 'b-')
        self.linePsi, = ax.plot(ini_arr,arrPsi, 'k-')

        plt.legend(["Z", "Phi", "Theta", "Psi"])
        plt.ylabel("Controlled Variables")
        plt.xlabel("Time")
        plt.grid(True)
        
        if(self.control_alg.__eq__("pd")):
            plt.savefig(self.fig_dir + "pd_plot.png")

    def update_plot(self, x, ep, save_fig=True):
        # Update and save plot
        x = np.array(x)
        lim_max = max([np.max(x[:,4]), np.max(x[:,6]), np.max(x[:,8]), np.max(x[:,10])])
        lim_min = min([np.min(x[:,4]), np.min(x[:,6]), np.min(x[:,8]), np.min(x[:,10])])

        self.fig.canvas.flush_events()
        plt.ylim((lim_min, lim_max))
        self.lineZ.set_ydata(x[:, 4])
        self.linePhi.set_ydata(x[:, 6])
        self.lineTheta.set_ydata(x[:, 8])
        self.linePsi.set_ydata(x[:, 10])
        self.fig.canvas.draw()
        if save_fig:
            plt.savefig(self.fig_dir + "episode_" + str(ep) + ".png")
    def save_episode(self,ep):
        # Save episode
        self.save_xlsx("episode_" + str(ep) + ".xlsx")
        self.model.save_weights(self.model_dir + "episode_" + str(ep) + ".hdf5")
  
    def save_xlsx(self,file_name=""):
        # save exel file
          xlsx = openpyxl.Workbook()
          sheet = xlsx.active
          for _,row in enumerate(self.x):
              row_ = row.tolist()
              sheet.append(row_)
          xlsx.save(self.xlsx_dir + file_name)
         
    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset simulation
        self.initialize_sim()
        self.time_step = 0

        self.state = np.array([0, 0, self.Zd-self.x0[4], self.x0[4],0]).reshape((5,))
        return self.state

    def build_model(self):
        states = self.observation_space.shape[0]
        n_actions = self.action_values.shape[0]

        # Build model
        input = Input(shape=(1,states+1))
        x1 = Dense(4)(input)
        x1 = Dense(400, activation="relu")(x1)
        x1 = Dense(300)(x1)
        x2 = Dense(1)(input)
        x2 = Dense(300)(x2)
        added = concatenate([x1, x2])
        relu = ReLU()(added)
        output = Dense(n_actions)(relu)
        output = Reshape((n_actions,))(output)

        model = Model(inputs=[input],outputs= [output])
        self.model = model

    def build_agent(self,mem_limit):
        # Build agent
        n_actions = self.action_values.shape[0]
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=mem_limit, window_length=1)
        dqn = DQNAgent(model=self.model, memory=memory, policy=policy,nb_actions=n_actions, nb_steps_warmup=5, target_model_update=0.9)
        return dqn

# Test the algorithms
control_alg = "dqn"
env = UAVEnv(delta_action=0.2,action_list=[-0.42,0.42],deltaT=0.01,tot_time=10,control_alg=control_alg)

if(control_alg.__eq__("dqn")):
    
    states = env.observation_space
    actions = env.action_space

    env.initialize_plot()
    env.build_model()

    n_episode = 5000

    dqn = env.build_agent(mem_limit=n_episode*env.total_time_step)
    dqn.compile(Adam(lr=1e-3), metrics=['mse','mae'])

    dqn.fit(env, nb_steps=n_episode*env.total_time_step, visualize=False, verbose=1,log_interval=env.total_time_step)

elif(control_alg.__eq__("pd")):
    for i in range(env.total_time_step):
        env.time_step_run()
        
    env.save_xlsx("pd_xlsx.xlsx")
    env.generate_plot()