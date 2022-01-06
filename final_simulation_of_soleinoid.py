# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:08:12 2021

@author: Mc Zie
"""

import pandas as pd
import numpy as np

import numpy as np
#We choose 5 cross validation for our machine learning model
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.utils import np_utils
from numpy import std
import seaborn as sns
import keras
import keras.utils
from keras import utils as np_utils
from tensorflow.keras.utils import to_categorical

from keras.layers import Dropout
# Example of Dropout on the Sonar Dataset: Hidden Layer
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD


import datetime
import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)
import csv
import adafruit_ads1x15.ads1015 as ADS
#import adafruit_ads1x15.ads1115 as ADS

from adafruit_ads1x15.analog_in import AnalogIn


slope = 1.48; #slope from linear fit
intercept = -1.56 # intercept from linear fit

ads = ADS.ADS1015(i2c)

chan = AnalogIn(ads, ADS.P0)
#voltage=chan.voltage

while True: 
    try: 
        voltage = round((chan.voltage),2)
        print( 'voltage:')
        print(f'{chan.voltage} Volt')
        
        vol_water_cont = ((1.0/chan.voltage)*slope)+intercept #calc of theta_v (vol. water content)
        vol_water_cont= round((vol_water_cont),2)
        print(" V, Theta_v: ")
        print(f'{vol_water_cont} cm^3/cm^3')
        
        if vol_water_cont>-0.40:
            print('wet')
        else: 
            
            print('dry')
        
        def get_voltage(): 
            
            voltage = round((chan.voltage),2) 
           
            voltage = str(voltage) 
            
            return(voltage)
        
        def get_humidity():
            
            humidity = vol_water_cont
            humidity= round((humidity),2)
            
            humidity = str(humidity) 
           
            return(humidity)
        
        def date_now():
           today = datetime.datetime.now().strftime("%Y-%m-%d")
           today = str(today)
           return(today)
        
        def time_now():
            
            now = datetime.datetime.now().strftime("%H:%M:%S")
            now = str(now)
            return(now)
        
        def get_humidity_binary():
            
            if vol_water_cont>-0.40:
                
                return str('wet')
            else: 
                return str('dry') 
            
        
        
        def write_to_csv():
            
            #the a is for append, if w for write is used then it overwrites the file
            with open('/home/pi/sensor_readings.csv', mode='a') as sensor_readings: 
                
                sensor_write = csv.writer(sensor_readings, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                write_to_log = sensor_write.writerow([date_now(),time_now(),get_voltage(), get_humidity(),get_humidity_binary()])
                return(write_to_log) 
            
        print( write_to_csv())
        
        from numpy import loadtxt
        from keras.models import load_model
 
       # load model
        model = load_model('/home/pi/model_rwet_dry.h5')
           
        colnames=['date','time','voltage','humidity','target']
        df=pd.read_csv("/home/pi/sensor_readings.csv",names=colnames, header=None)
        
        
        cat_columns = df.select_dtypes(['object']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
        
        X=df[['voltage', 'humidity', 'date', 'time']]
        y=df['target']
        
        result = model.predict(X)
        

        
        def soleinoi_off_on():
            
            if result[1]>0.5: 
                
                print( 'soleinoid valve is off' )
            else: 
                
                print('soleinoid valve is on')
        
        time.sleep(1) 
        
    except KeyboardInterrupt: 
        break 
    except IOError: 
        print ("Error") 

