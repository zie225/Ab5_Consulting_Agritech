# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:21:49 2021

@author: Mc Zie
"""

import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)

import adafruit_ads1x15.ads1015 as ADS
#import adafruit_ads1x15.ads1115 as ADS

from adafruit_ads1x15.analog_in import AnalogIn

ads = ADS.ADS1015(i2c)

chan = AnalogIn(ads, ADS.P0)


while True: 
    try: 
        print( 'voltage:')
        print( chan.voltage)
        
        time.sleep(.5) 
 
    except KeyboardInterrupt: 
        break 
    except IOError: 
        print ("Error") 
