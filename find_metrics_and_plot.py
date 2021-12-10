# -*- coding: utf-8 -*-


from tkinter import filedialog 
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# root = tk.Tk()
# root.withdraw()

# file_path_dqn = tk.filedialog.askopenfilenames()[0]
# file_path_pd  = tk.filedialog.askopenfilenames()[0]

file_path_dqn = "episode_29.xlsx"
file_path_pd  = "pd_xlsx.xlsx"

df_dqn = pd.read_excel(file_path_dqn,header=None)
df_pd  = pd.read_excel(file_path_pd,header=None)


z_altitude_dqn = df_dqn[4].to_numpy()
z_altitude_pd = df_pd[4].to_numpy()

phi_pd            = df_dqn[6].to_numpy()
theta_pd          = df_dqn[8].to_numpy()
psi_pd            = df_dqn[10].to_numpy()

deltaT = 0.01
ref = 0.5

fig = plt.figure()

ax = fig.add_subplot(111)
ini_arr = np.arange(z_altitude_dqn.shape[0])
ini_arr = np.multiply(ini_arr, deltaT)
ref_arr = np.ones(ini_arr.shape[0])*ref

line_ref, = ax.plot(ini_arr,ref_arr , 'k-')
lineZ_dqn, = ax.plot(ini_arr, z_altitude_dqn, 'b-')
lineZ_pd, = ax.plot(ini_arr, z_altitude_pd, 'r-')

lim_max = max([np.max(ref_arr), np.max(z_altitude_dqn), np.max(z_altitude_pd)])
lim_min = min([np.min(ref_arr), np.min(z_altitude_dqn), np.min(z_altitude_pd)])
plt.ylim((lim_min, lim_max+lim_max*0.1))
plt.xlim((0,10))
  

plt.legend(["Reference", "DQN", "PD"])
plt.ylabel("Z-Altitude(m)")
plt.xlabel("Time(s)")
# plt.grid(True)
plt.show()

# Phi Angle
fig = plt.figure()

ax = fig.add_subplot(111)
ini_arr = np.arange(phi_pd.shape[0])
ini_arr = np.multiply(ini_arr, deltaT)
ref_arr = np.ones(ini_arr.shape[0])*ref

line_ref, = ax.plot(ini_arr,ref_arr , 'k-')
linePhi_pd, = ax.plot(ini_arr, phi_pd, 'b-')

lim_max = max([np.max(ref_arr), np.max(phi_pd)])
lim_min = min([np.min(ref_arr), np.min(phi_pd)])
plt.ylim((lim_min, lim_max+lim_max*0.1))
plt.xlim((0,10))
  

plt.legend(["Reference", "PD"])
plt.ylabel("Phi Angle(rad)")
plt.xlabel("Time(s)")
# plt.grid(True)
plt.show()

# Theta
fig = plt.figure()

ax = fig.add_subplot(111)
ini_arr = np.arange(theta_pd.shape[0])
ini_arr = np.multiply(ini_arr, deltaT)
ref_arr = np.ones(ini_arr.shape[0])*ref

line_ref, = ax.plot(ini_arr,ref_arr , 'k-')
lineTheta_pd, = ax.plot(ini_arr, theta_pd, 'b-')

lim_max = max([np.max(ref_arr), np.max(theta_pd)])
lim_min = min([np.min(ref_arr), np.min(theta_pd)])
plt.ylim((lim_min, lim_max+lim_max*0.1))
plt.xlim((0,10))
  

plt.legend(["Reference", "PD"])
plt.ylabel("Theta Angle(rad)")
plt.xlabel("Time(s)")
# plt.grid(True)
plt.show()

# Psi
fig = plt.figure()

ax = fig.add_subplot(111)
ini_arr = np.arange(psi_pd.shape[0])
ini_arr = np.multiply(ini_arr, deltaT)
ref_arr = np.ones(ini_arr.shape[0])*ref

line_ref, = ax.plot(ini_arr,ref_arr , 'k-')
linePsi_pd, = ax.plot(ini_arr, psi_pd, 'b-')

lim_max = max([np.max(ref_arr), np.max(psi_pd)])
lim_min = min([np.min(ref_arr), np.min(psi_pd)])
plt.ylim((lim_min, lim_max+lim_max*0.1))
plt.xlim((0,10))
  

plt.legend(["Reference", "PD"])
plt.ylabel("Psi Angle(rad)")
plt.xlabel("Time(s)")
# plt.grid(True)
plt.show()


# For Z altitude (pd)
error_z_pd   = ref - z_altitude_pd
error_z_pd_2 = np.multiply(error_z_pd,error_z_pd)

MSE_z_pd     = np.mean(error_z_pd_2)
ISE_z_pd     = np.sum(error_z_pd_2)
IAE_z_pd     = np.sum(np.abs(error_z_pd))

# For Z altitude (dqn)
error_z_dqn   = ref - z_altitude_dqn
error_z_dqn_2 = np.multiply(error_z_dqn,error_z_dqn)

MSE_z_dqn     = np.mean(error_z_dqn_2)
ISE_z_dqn     = np.sum(error_z_dqn_2)
IAE_z_dqn     = np.sum(np.abs(error_z_dqn))

# For Phi angle (pd)
error_phi_pd   = ref - phi_pd
error_phi_pd_2 = np.multiply(error_phi_pd,error_phi_pd)

MSE_phi_pd     = np.mean(error_phi_pd_2)
ISE_phi_pd     = np.sum(error_phi_pd_2)
IAE_phi_pd     = np.sum(np.abs(error_phi_pd))

# For Theta angle (pd)
error_theta_pd   = ref - theta_pd
error_theta_pd_2 = np.multiply(error_theta_pd,error_theta_pd)

MSE_theta_pd     = np.mean(error_theta_pd_2)
ISE_theta_pd     = np.sum(error_theta_pd_2)
IAE_theta_pd     = np.sum(np.abs(error_theta_pd))

# For Psi angle (pd)
error_psi_pd   = ref - psi_pd
error_psi_pd_2 = np.multiply(error_psi_pd,error_psi_pd)

MSE_psi_pd     = np.mean(error_psi_pd_2)
ISE_psi_pd     = np.sum(error_psi_pd_2)
IAE_psi_pd     = np.sum(np.abs(error_psi_pd))

print(f"DQN for Z : \nMean Square Error: {MSE_z_dqn} \nIntegral Square Error: {ISE_z_dqn} \nIntegral Absolute Error: {IAE_z_dqn}\n\n")
print(f"PD for Z : \nMean Square Error: {MSE_z_pd} \nIntegral Square Error: {ISE_z_pd} \nIntegral Absolute Error: {IAE_z_pd}\n\n")


print(f"PD for Phi: \nMean Square Error: {MSE_phi_pd} \nIntegral Square Error: {ISE_phi_pd} \nIntegral Absolute Error: {IAE_phi_pd}\n\n")
print(f"PD for Theta: \nMean Square Error: {MSE_theta_pd} \nIntegral Square Error: {ISE_theta_pd} \nIntegral Absolute Error: {IAE_theta_pd}\n\n")
print(f"PD for Psi: \nMean Square Error: {MSE_psi_pd} \nIntegral Square Error: {ISE_psi_pd} \nIntegral Absolute Error: {IAE_psi_pd}\n\n")

# Find Setting Time
setting_time_z_dqn    = (np.max(np.where(z_altitude_dqn<ref*0.98))+1)/100
setting_time_z_pd     = (np.max(np.where(z_altitude_pd<ref*0.98))+1)/100
setting_time_phi_pd   = (np.max(np.where(phi_pd<ref*0.98))+1)/100
setting_time_theta_pd = (np.max(np.where(theta_pd<ref*0.98))+1)/100
setting_time_psi_pd   = (np.max(np.where(psi_pd<ref*0.98))+1)/100

print(f"Setting Time for Z DQN  : {setting_time_z_dqn}")
print(f"Setting Time for Z PD   : {setting_time_z_pd}")
print(f"Setting Time for phi PD : {setting_time_phi_pd}")
print(f"Setting Time for theta PD : {setting_time_theta_pd}")
print(f"Setting Time for psi PD : {setting_time_psi_pd}\n\n")

# Find Overshoot 
overshoot_z_dqn     = (np.max(z_altitude_dqn)-ref)*100/ref
overshoot_z_pd      = (np.max(z_altitude_pd)-ref)*100/ref
overshoot_phi_pd    = (np.max(phi_pd)-ref)*100/ref
overshoot_theta_pd  = (np.max(theta_pd)-ref)*100/ref
overshoot_psi_pd    = (np.max(psi_pd)-ref)*100/ref

print(f"Overshoot for Z DQN  : {overshoot_z_dqn}")
print(f"Overshoot for Z PD   : {overshoot_z_pd}")
print(f"Overshoot for phi PD : {overshoot_phi_pd}")
print(f"Overshoot for theta PD : {overshoot_theta_pd}")
print(f"Overshoot for psi PD : {overshoot_psi_pd}")
