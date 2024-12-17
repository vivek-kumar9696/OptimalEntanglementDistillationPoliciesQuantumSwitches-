from zeroth_iteration_switchModel import *
from ZerothSwitchEnv import * 


import gymnasium as gym
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

import os
from multiprocessing import Pool


fid_thresh_list_distill = [(True,x / 100) for x in range(70, 101)] # creating list of fidelity thresholds for distillation from 0.7 to 1 at increments of 0.01
fid_thresh_list_noDistill = [(False,x / 100) for x in range(70, 101)] # creating list of fidelity thresholds for only swap from 0.7 to 1 at increments of 0.01

fid_thresh_list = fid_thresh_list_distill + fid_thresh_list_noDistill


def custom_env_creator(param1, param2, param3, param4, param5):
    return ZerothSwitchEnv(switch_config_file=param1, switch_name=param2, fid_thresh=param3, distillFlag=param4, x_value=param5)



def model_train(distillFlag,fid_thresh):
   
    x_list=[0]#, 0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1.25,1.375,1.5,1.625,1.75,1.875,2]
    train_ts_list=[125000]#, 10000, 10000, 10000, 10000, 10000, 7500, 7500, 7500, 7500, 5000, 5000, 5000, 5000, 2000, 2000]
    total_ts = sum(train_ts_list)
    
    models_dir= f"models/distill_allowed_{distillFlag}/{fid_thresh}/"
    logdir= f"logs/distill_allowed_{distillFlag}/{fid_thresh}/"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    
    curriculum = ["ZerothSwitch_"+str(fid_thresh)+"_distillationAllowed_"+str(distillFlag)+"_curriculum_x"+str(x)+"-v0" for x in x_list]
    

    for index,env_id in enumerate(curriculum):
        gym.register(
        id=env_id,
        entry_point='train_model:custom_env_creator',
        kwargs={'param1': "switch_config.ini", 'param2': "S1",'param3':fid_thresh,'param4':distillFlag,'param5':x_list[index]}
    )
        
        env = gym.make(env_id)
        
        if index == 0:
            model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, tensorboard_log=logdir)
        else:
            model.set_env(env)
    
        model.learn(total_timesteps=train_ts_list[index], reset_num_timesteps=False, tb_log_name="PPO")
        
    model.save(f"{models_dir}/{total_ts}_{fid_thresh}")
    


def multiprocessed_train():
      
    with Pool() as executor:
        results_distillation_allowed = executor.starmap(model_train, fid_thresh_list)
        

    
    
    
    

