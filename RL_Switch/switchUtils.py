# utils.py

import math
import configparser
import gymnasium as gym
import numpy as np


def distillation_psucc_fidelity(F_link1, F_link2):
    
    p_succ = ((8/9)*(F_link1*F_link2))-((2/9)*(F_link1 + F_link2)) + (5/9)
    
    resultant_fidelity = (((10/9)*(F_link1*F_link2))-((1/9)*(F_link1 + F_link2)) + (1/9))/(p_succ)
    return resultant_fidelity, p_succ

def ES_fidelity(F1, F2):
    resultant_fidelity = (((4*F1 - 1) * (4*F2 - 1) / 3) + 1) / 4
    
    return resultant_fidelity


def fid_cap_reward_stepFunction(fid, fid_thresh):
    
    if fid < fid_thresh:
        return 0
    
    else:
        return 1
    
def fid_cap_reward_cont(fid, fid_thresh):
    
    if fid < fid_thresh:
        return 0
    
    else:
        return fid 
    
def custom_sort_key(item):
    return int(item)

def s_curve(l , x):
    return 1 / (1 + np.exp(-l * (1 - x)))


def novel_feature(user_list, fid_thresh):
    novel_feature_list = []
    for i, lst in enumerate(user_list):
        for j, val in enumerate(lst):
            count = 0
            for k, other_lst in enumerate(user_list):
                if k != i:  # Exclude the current list
                    for other_val in other_lst:
                        if val * other_val >= fid_thresh:
                            count += 1
            novel_feature_list.append(count)
    novel_feature_string = ''.join(map(str, novel_feature_list))
    return novel_feature_string

    
        