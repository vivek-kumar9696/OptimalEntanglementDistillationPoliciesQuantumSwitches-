from sb3_contrib.ppo_mask import MaskablePPO
from zeroth_iteration_switchModel import *
from ZerothSwitchEnv import * 

import math
from multiprocessing import Pool
import json

seed = 42  # Set your desired seed

mean_reward_distill = {}
mean_reward_onlySwap = {}


TIMESTEPS =50000

fid_thresh_list_distill = [(True,x / 100) for x in range(70, 101)] # creating list of fidelity thresholds for distillation from 0.7 to 1 at increments of 0.01
fid_thresh_list_noDistill = [(False,x / 100) for x in range(70, 101)] # creating list of fidelity thresholds for only swap from 0.7 to 1 at increments of 0.01

timesteps_sim = 10000
num_episodes = 5
warmup_time = 1000

def simulateSwitch(distillFlag,fid_thresh):
    
    mean_reward = ()
    episode_mean_rewards = []
    env = ZerothSwitchEnv("switch_config.ini", "S1", fid_thresh, distillFlag)
    
    models_dir = f"models/distill_allowed_{distillFlag}/{fid_thresh}/{TIMESTEPS}_{fid_thresh}"
    model = MaskablePPO.load(models_dir, env=env)
    model.set_random_seed(seed)
    
    e = 0
    
    while e < num_episodes:
        obs, _ = env.reset() 
        episode_reward = 0
        current_time = 0
        while current_time < timesteps_sim:  # Run for the specified number of timesteps
            action, _ = model.predict(obs,action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            if current_time  > warmup_time:
                episode_reward += reward
            else:
                pass
            current_time += 1
        episode_mean_rewards.append(episode_reward/timesteps_sim)
        e += 1
    env.close()
    mean_reward = (fid_thresh,sum(episode_mean_rewards) / num_episodes)
    return mean_reward


    
def multiprocessed_sim():
    with Pool() as executor:
        results_distillation_allowed = executor.starmap(simulateSwitch, fid_thresh_list_distill)
        results_distillation_notAllowed = executor.starmap(simulateSwitch, fid_thresh_list_noDistill)
    
    # Create dictionary
    for tuple in results_distillation_allowed:
        mean_reward_distill[tuple[0]] = tuple[1]
    for tuple in results_distillation_notAllowed:
        mean_reward_onlySwap[tuple[0]] = tuple[1]
        
    
    with open('predict_distill_allowed_True.txt', 'w') as file:
        json.dump(mean_reward_distill, file)
    with open('predict_distill_allowed_False.txt', 'w') as file:
        json.dump(mean_reward_onlySwap, file)
        
    print("distill allowed:", mean_reward_distill)
    print("distill not allowed:", mean_reward_onlySwap)


        
    
    
