import time
import numpy as np
from datetime import datetime



def calculate_velocity(pre_pos, cur_pos, pre_time, cur_time):
    return (cur_pos - pre_pos) / (cur_time - pre_time)


initial_position = np.array([0,0,0])
initial_time= float(datetime.now().strftime('%Y%m%d%H%M%S.%f'))


while True: 
    current_position = np.array([1,1,1])
    current_time = float(datetime.now().strftime('%Y%m%d%H%M%S'))
    print(calculate_velocity(initial_position, current_position, initial_time, current_time))
    initial_position = current_position
    initial_time = current_time


