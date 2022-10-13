import numpy as np

##Experiment info
psychopyVersion = '3.0.1'
expName = 'No-Adaptation'
expInfo = {'participant': '', 'session': '', 'vf' : ''}

##Window
windowSize = [1920, 1200]

##Trial number and staircase info
numTrials = 25
start_angle_sc1 = 15.0
start_angle_sc2 = -10.0

sc_states = [
    [[2.5, 0, 1, 'cN2'], [start_angle_sc1], [], 0],
    [[2.5, 0, 2, 'cN2'], [start_angle_sc2], [], 0],
    [[6.5, 0, 1, 'cN1'], [start_angle_sc1], [], 0],
    [[6.5, 0, 2, 'cN1'], [start_angle_sc2], [], 0],
    [[10.5, 0, 1, 'cC'], [start_angle_sc1], [], 0],
    [[10.5, 0, 2, 'cC'], [start_angle_sc2], [], 0],
    [[14.5, 0, 1, 'cF1'], [start_angle_sc1], [], 0],
    [[14.5, 0, 2, 'cF1'], [start_angle_sc2], [], 0],
    [[18.5, 0, 1, 'cF2'], [start_angle_sc1], [], 0],
    [[18.5, 0, 2, 'cF2'], [start_angle_sc2], [], 0],
    [[2.5, -4, 1, 'dN2'], [start_angle_sc1], [], 0],
    [[2.5, -4, 2, 'dN2'], [start_angle_sc2], [], 0],
    [[6.5, -4, 1, 'dN1'], [start_angle_sc1], [], 0],
    [[6.5, -4, 2, 'dN1'], [start_angle_sc2], [], 0],
    [[10.5, -4, 1, 'dC'], [start_angle_sc1], [], 0],
    [[10.5, -4, 2, 'dC'], [start_angle_sc2], [], 0],
    [[14.5, -4, 1, 'dF1'], [start_angle_sc1], [], 0],
    [[14.5, -4, 2, 'dF1'], [start_angle_sc2], [], 0],
    [[18.5, -4, 1, 'dF2'], [start_angle_sc1], [], 0],
    [[18.5, -4, 2, 'dF2'], [start_angle_sc2], [], 0],
    [[2.5, 4, 1, 'uN2'], [start_angle_sc1], [], 0],
    [[2.5, 4, 2, 'uN2'], [start_angle_sc2], [], 0],
    [[6.5, 4, 1, 'uN1'], [start_angle_sc1], [], 0],
    [[6.5, 4, 2, 'uN1'], [start_angle_sc2], [], 0],
    [[10.5, 4, 1, 'uC'], [start_angle_sc1], [], 0],
    [[10.5, 4, 2, 'uC'], [start_angle_sc2], [], 0],
    [[14.5, 4, 1, 'uF1'], [start_angle_sc1], [], 0],
    [[14.5, 4, 2, 'uF1'], [start_angle_sc2], [], 0],
    [[18.5, 4, 1, 'uF2'], [start_angle_sc1], [], 0],
    [[18.5, 4, 2, 'uF2'], [start_angle_sc2], [], 0]
    ]

location_ind_list = np.arange(0, 25)

