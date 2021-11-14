# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:00:34 2021

@author: Mc Zie
"""

import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)

import adafruit_ads1x15.ads1015 as ADS
#import adafruit_ads1x15.ads1115 as ADS

from adafruit_ads1x15.analog_in import AnalogIn


slope = 1.48; #slope from linear fit
intercept = -1.56 # intercept from linear fit

ads = ADS.ADS1015(i2c)

chan = AnalogIn(ads, ADS.P0)
#voltage=chan.voltage
#vol_water_cont = ((1.0/chan.voltage)*slope)+intercept #calc of theta_v (vol. water content)

while True: 
    try: 
        print( 'voltage:')
        print(f'{chan.voltage} Volt')
        
        vol_water_cont = ((1.0/chan.voltage)*slope)+intercept #calc of theta_v (vol. water content)

        print(" V, Theta_v: ")
        print(f'{vol_water_cont} cm^3/cm^3')
        
        time.sleep(.5) 
 
    except KeyboardInterrupt: 
        break 
    except IOError: 
        print ("Error") 