# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:35:41 2021

@author: Mc Zie
"""

import datetime
import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)
import csv
import adafruit_ads1x15.ads1015 as ADS
#import adafruit_ads1x15.ads1115 as ADS
from w1thermsensor import W1ThermSensor
from adafruit_ads1x15.analog_in import AnalogIn
import RPi.GPIO as GPIO

ds18b20 = W1ThermSensor()

ads = ADS.ADS1015(i2c)
ads.gain = 1

interval = 5 #How long we want to wait between loops (seconds)
waterTick = 0 #Used to count the number of times the flow input is triggered

import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection
import sklearn as read_csv
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from random import shuffle
from operator import itemgetter
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import model_selection
import sklearn as read_csv
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
from random import shuffle
from operator import itemgetter
from sklearn import preprocessing
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.random import check_random_state
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from matplotlib import pyplot
#We choose 5 cross validation for our machine learning model
n_folds = KFold(n_splits =5,shuffle= False )
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import jaccard_score
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold # evaluate a given model using cross-validation

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import random


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
        
  ############## ARTIFICIAL INTELLIGENCE APPLY IN SOLEINOID VALVE ############################
        from numpy import loadtxt
        from keras.models import load_model
 
        # load model Neural network
        model = load_model('model_rwet_dry.h5')
        
        colnames=['date','time','voltage','humidity','target']
        
        df1=pd.read_csv('/home/pi/sensor_readings.csv',names=colnames, header=None)
        
        encoded=df1[['date','time','target']].apply(LabelEncoder().fit_transform)
        remain=df1[['voltage','humidity']]
        
        # Adding both the dataframes encoded and remaining (without encoding)
        data =pd.concat([remain,encoded], axis=1)
        
        X=data[['voltage', 'humidity', 'date', 'time']]
        y=data['target']
        
        result = model.predict(X)
         
        #Convert result predict in binary
        from sklearn.preprocessing import binarize

 
        predict = np.ravel(binarize(result.reshape(-1,1), 0.5))
        
        if predict[0]>0.5: 
            print("The soil is wet") 
        else: 
                
            print("The soil is dry")
         ### SOLEINOID DATA AND READER ######################################
         
        ds18b20 = W1ThermSensor()

        ads = ADS.ADS1015(i2c)
        ads.gain = 1

        interval = 1 #How long we want to wait between loops (seconds)
        waterTick = 0 #Used to count the number of times the flow input is triggered

#Assign channels to variables to keep track of them easier (these BCM pin numbers were listed in part 1 of the tutorial)
        s1 = 13
        s2 = 16
        s3 = 19
        s4 = 20
        s5 = 26
        s6 = 21

#Set GPIO pins to use BCM pin numbers
        GPIO.setmode(GPIO.BCM)

#Set digital pin 24 to an input
        GPIO.setup(24, GPIO.IN)

#Set solenoid driver pins to outputs:
        GPIO.setup(s1, GPIO.OUT) #set Solenoid 1 output
        GPIO.setup(s2, GPIO.OUT) #set Solenoid 2 output
        GPIO.setup(s3, GPIO.OUT) #set Solenoid 3 output
        GPIO.setup(s4, GPIO.OUT) #set Solenoid 4 output
        GPIO.setup(s5, GPIO.OUT) #set Solenoid 5 output
        GPIO.setup(s6, GPIO.OUT) #set Solenoid 6 output

#Event to detect flow (1 tick per revolution)
        GPIO.add_event_detect(24, GPIO.FALLING)
        def flowtrig(self):  
            
            global waterTick  
                
            waterTick += 1

        GPIO.add_event_callback(24, flowtrig)

        while True: 
            
            time.sleep(interval)

#Pull Temperature from DS18B20
            #temperature = ds18b20.get_temperature()

#Measure Analog Input 0
        chan = AnalogIn(ads, ADS.P0) #ADS.P1 , P2, P3 for channels 1, 2, 3
            #val = chan.value #Pull the raw ADC data from Channel 0

        waterFlow = waterTick * 2.25
        waterTick = 0

#Test Solenoids by turning each on for a 1 second
        GPIO.output(s1, GPIO.HIGH) #turn solenoid 1 on
        time.sleep(1) #Wait for a half second
        GPIO.output(s1, GPIO.LOW) #turn solenoid 1 off
        time.sleep(1)
            
        def soleinoi_off_on():  
            
            if predict[0]>0.5: 
                
                return  GPIO.output(s1, GPIO.LOW) #turn solenoid 1 off
            
                time.sleep(1)
            else: 
                return  GPIO.output(s1, GPIO.HIGH) #turn solenoid 1 on
            
                time.sleep(1)
                
                print(soleinoi_off_on())
            
            """
            GPIO.output(s2, GPIO.HIGH) #turn solenoid 2 on
            time.sleep(0.5) #Wait for a half second
            GPIO.output(s2, GPIO.LOW) #turn solenoid 2 off
            time.sleep(0.5)

            GPIO.output(s3, GPIO.HIGH) #turn solenoid 3 on
            time.sleep(0.5) #Wait for a half second
            GPIO.output(s3, GPIO.LOW) #turn solenoid 3 off
            time.sleep(0.5)

            GPIO.output(s4, GPIO.HIGH) #turn solenoid 4 on
            time.sleep(0.5) #Wait for a half second
            GPIO.output(s4, GPIO.LOW) #turn solenoid 4 off
            time.sleep(0.5)

            GPIO.output(s5, GPIO.HIGH) #turn solenoid 5 on
            time.sleep(0.5) #Wait for a half second
            GPIO.output(s5, GPIO.LOW) #turn solenoid 5 off
            time.sleep(0.5)

            GPIO.output(s6, GPIO.HIGH) #turn solenoid 6 on
            time.sleep(0.5) #Wait for a half second
            GPIO.output(s6, GPIO.LOW) #turn solenoid 6 off
            time.sleep(0.5)

#Print the results
   
            print( 'Soil Moisture:' , val)
            print( 'Flow Rate:' , waterFlow)
            print( " ")
        
            """
        
        
        time.sleep(1) 
        
    except KeyboardInterrupt: 
        break 
    except IOError: 
        print ("Error") 

