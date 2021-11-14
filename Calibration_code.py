# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 23:58:04 2021

@author: Mc Zie

"""

##############################
# Capacitive Soil Moisture
# Sensor Calibration Analysis
##############################
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

container_mass = 19.67 # measured mass of container [g]
soil_mass_dry = 183 # mass of dry soil [g]
soil_vol = 153 # volume of soil sample [ml]

rho_s = (soil_mass_dry/1000.0)/(soil_vol*np.power(10.0,-6.0)) # bulk density of soil [kg/m^3]
rho_w = 997.0 # density of water [kg/m^3]

###############################################
# Data inputs
#
soil_masses = np.subtract([116.45,137.68,151.69,170.22,183.56,
                           191.14,193.31,194.48,197.19],container_mass) # mass measurements [g]
cap_sensor_readings = np.array([1.63,1.43,1.36,1.32,1.29,1.26,1.25,1.20,1.19]) # cap sensor readings [V]

###############################################
# calculating volumetric water content [%]
#
theta_g = (soil_masses - soil_mass_dry)/soil_mass_dry # water proportion
theta_v = ((theta_g*rho_s)/rho_w) # volumetric soil content [g/ml / g/ml]

###############################################
# Fitting 1/sensor readings with measurements
#
x_for_training = 1.0/cap_sensor_readings # 1/sensor readings
slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_training, theta_v) # linear fit
theta_predict = (slope*(x_for_training))+intercept # prediction of theta_v with sensor

###############################################
# Plot the results
#


plt.style.use('ggplot')
fig,axs = plt.subplots(2,1,figsize=(12,9))
# plotting the sensor to theta_v
ax = axs[0]
fig.suptitle('Capacitive soil moisture sensor calibration  for sandy soil', fontsize=16)

ax.plot(x_for_training,theta_v,label='Data',linestyle='',marker='o',color=plt.cm.Set1(0),
       markersize=10,zorder=999)
ax.plot(x_for_training,theta_predict,label='Fit ({0:2.2f}$\cdot$(1/V) {1:+2.2f})'.format(slope,intercept),
        color=plt.cm.Set1(1),linewidth=4)
ax.set_xlabel(r'Inverse of Capacitive Sensor Voltage [V$^{-1}$]',fontsize=18)
ax.set_ylabel(r'$\theta_v$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax.legend(fontsize=16)

rmse = np.sqrt(np.mean(np.power(np.subtract(theta_predict,theta_v),2.0))) # value error
mape = np.mean(np.divide(np.subtract(theta_predict,theta_v),theta_v)*100) # % error

# plotting the comparison between fit and data
ax2 = axs[1]
ax2.plot(theta_predict,theta_v,label='Capacitive (RMSE: {0:2.3f}, MAPE: {1:2.0f}%)'.format(rmse,mape),
         linestyle='',marker='o',color=plt.cm.Set1(2),markersize=10,zorder=999)
ax2.plot(theta_v,theta_v,label='Gravimetric',color=plt.cm.Set1(3),linewidth=4)
ax2.set_xlabel(r'$\theta_{v,cap}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.set_ylabel(r'$\theta_{v,grav}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.legend(fontsize=16)
fig.savefig('soil_moisture_calibration_results_for_sandy_soil.png',dpi=300,bbox_inches='tight',facecolor='#FCFCFC')
plt.show()

#################################################################################################################################"

##############################
# Capacitive Soil Moisture
# Sensor Calibration Analysis
##############################
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

container_mass = 19.67 # measured mass of container [g]
soil_mass_dry = 183 # mass of dry soil [g]
soil_vol = 153 # volume of soil sample [ml]

rho_s = (soil_mass_dry/1000.0)/(soil_vol*np.power(10.0,-6.0)) # bulk density of soil [kg/m^3]
rho_w = 997.0 # density of water [kg/m^3]

###############################################
# Data inputs
#
soil_masses = np.subtract([102.23,112.92,133.81,139.65,145.12,
                           148.80,156.69,162.61,190.81],container_mass) # mass measurements [g]
cap_sensor_readings = np.array([1.47,1.40,1.32,1.27,1.25,1.24,1.22,1.21,1.20]) # cap sensor readings [V]

###############################################
# calculating volumetric water content [%]
#
theta_g = (soil_masses - soil_mass_dry)/soil_mass_dry # water proportion
theta_v = ((theta_g*rho_s)/rho_w) # volumetric soil content [g/ml / g/ml]

###############################################
# Fitting 1/sensor readings with measurements
#
x_for_training = 1.0/cap_sensor_readings # 1/sensor readings
slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_training, theta_v) # linear fit
theta_predict = (slope*(x_for_training))+intercept # prediction of theta_v with sensor

###############################################
# Plot the results
#


plt.style.use('ggplot')
fig,axs = plt.subplots(2,1,figsize=(12,9))
# plotting the sensor to theta_v
ax = axs[0]
fig.suptitle('Capacitive soil moisture sensor calibration  for clayey soil', fontsize=16)

ax.plot(x_for_training,theta_v,label='Data',linestyle='',marker='o',color=plt.cm.Set1(0),
       markersize=10,zorder=999)
ax.plot(x_for_training,theta_predict,label='Fit ({0:2.2f}$\cdot$(1/V) {1:+2.2f})'.format(slope,intercept),
        color=plt.cm.Set1(1),linewidth=4)
ax.set_xlabel(r'Inverse of Capacitive Sensor Voltage [V$^{-1}$]',fontsize=18)
ax.set_ylabel(r'$\theta_v$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax.legend(fontsize=16)

rmse = np.sqrt(np.mean(np.power(np.subtract(theta_predict,theta_v),2.0))) # value error
mape = np.mean(np.divide(np.subtract(theta_predict,theta_v),theta_v)*100) # % error

# plotting the comparison between fit and data
ax2 = axs[1]
ax2.plot(theta_predict,theta_v,label='Capacitive (RMSE: {0:2.3f}, MAPE: {1:2.0f}%)'.format(rmse,mape),
         linestyle='',marker='o',color=plt.cm.Set1(2),markersize=10,zorder=999)
ax2.plot(theta_v,theta_v,label='Gravimetric',color=plt.cm.Set1(3),linewidth=4)
ax2.set_xlabel(r'$\theta_{v,cap}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.set_ylabel(r'$\theta_{v,grav}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.legend(fontsize=16)
fig.savefig('soil_moisture_calibration_results_for_clayey_soil.png',dpi=300,bbox_inches='tight',facecolor='#FCFCFC')
plt.show()

########################################################################################################################

##############################
# Capacitive Soil Moisture
# Sensor Calibration Analysis
##############################
#
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

container_mass = 19.67 # measured mass of container [g]
soil_mass_dry = 183 # mass of dry soil [g]
soil_vol = 153 # volume of soil sample [ml]

rho_s = (soil_mass_dry/1000.0)/(soil_vol*np.power(10.0,-6.0)) # bulk density of soil [kg/m^3]
rho_w = 997.0 # density of water [kg/m^3]

###############################################
# Data inputs
#
soil_masses = np.subtract([61.30,88.71,109.00,123.92,131.80,
                           136.34,139.32,149.99,156.34],container_mass) # mass measurements [g]
cap_sensor_readings = np.array([2.17,1.99,1.53,1.39,1.33,1.30,1.26,1.24,1.21]) # cap sensor readings [V]

###############################################
# calculating volumetric water content [%]
#
theta_g = (soil_masses - soil_mass_dry)/soil_mass_dry # water proportion
theta_v = ((theta_g*rho_s)/rho_w) # volumetric soil content [g/ml / g/ml]

###############################################
# Fitting 1/sensor readings with measurements
#
x_for_training = 1.0/cap_sensor_readings # 1/sensor readings
slope, intercept, r_value, p_value, std_err = stats.linregress(x_for_training, theta_v) # linear fit
theta_predict = (slope*(x_for_training))+intercept # prediction of theta_v with sensor

###############################################
# Plot the results
#


plt.style.use('ggplot')
fig,axs = plt.subplots(2,1,figsize=(12,9))
# plotting the sensor to theta_v
ax = axs[0]
fig.suptitle('Capacitive soil moisture sensor calibration  for silty soil', fontsize=16)

ax.plot(x_for_training,theta_v,label='Data',linestyle='',marker='o',color=plt.cm.Set1(0),
       markersize=10,zorder=999)
ax.plot(x_for_training,theta_predict,label='Fit ({0:2.2f}$\cdot$(1/V) {1:+2.2f})'.format(slope,intercept),
        color=plt.cm.Set1(1),linewidth=4)
ax.set_xlabel(r'Inverse of Capacitive Sensor Voltage [V$^{-1}$]',fontsize=18)
ax.set_ylabel(r'$\theta_v$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax.legend(fontsize=16)

rmse = np.sqrt(np.mean(np.power(np.subtract(theta_predict,theta_v),2.0))) # value error
mape = np.mean(np.divide(np.subtract(theta_predict,theta_v),theta_v)*100) # % error

# plotting the comparison between fit and data
ax2 = axs[1]
ax2.plot(theta_predict,theta_v,label='Capacitive (RMSE: {0:2.3f}, MAPE: {1:2.0f}%)'.format(rmse,mape),
         linestyle='',marker='o',color=plt.cm.Set1(2),markersize=10,zorder=999)
ax2.plot(theta_v,theta_v,label='Gravimetric',color=plt.cm.Set1(3),linewidth=4)
ax2.set_xlabel(r'$\theta_{v,cap}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.set_ylabel(r'$\theta_{v,grav}$ [cm$^{3}$/cm$^3$]',fontsize=18)
ax2.legend(fontsize=16)
fig.savefig('soil_moisture_calibration_results_for_silty_soil.png',dpi=300,bbox_inches='tight',facecolor='#FCFCFC')
plt.show()