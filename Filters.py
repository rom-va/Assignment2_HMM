
import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
# By Romina and Sahand
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm # Transition model
        self.__om = om # Observation model
        self.__sm = sm # State model
        self.__f = probs # Probability vector up to the previous step (stores f_1:i-1)
        
    # sensorR is the sensor reading (index!), self._f is the probability distribution resulting from the filtering    
    def filter(self, sensorR : int) -> np.array : # Takes the sensor reading in the form of an index
        # Implementation of simple forward filtering algorithm for one sensor reading (finding f_1:i)
        # If the reading is None
        #if sensorR is not None:
        O_i = self.__om.get_o_reading(sensorR)
        
        T_matrix = self.__tm.get_T()
        
        f_updated = O_i @ np.transpose(T_matrix) @ self.__f
        f_updated = f_updated / np.sum(f_updated)
        self.__f = f_updated
        
        return self.__f


class HMMSmoother:
    
    def __init__(self, tm, om, sm):
        self.__tm = tm
        self.__om = om
        self.__sm = sm

    # sensor_r_seq is the sequence (array) with the t-k sensor readings for smoothing, 
    # f_k is the filtered result (f_vector) for step k
    # fb is the smoothed result (fb_vector)
    def smooth(self, sensor_r_seq : np.array, f_k : np.array) -> np.array:
        fb = self.__f # setting a dummy value here...
        # somehow compute fb to be better than f ;-)
        # ...
        return fb