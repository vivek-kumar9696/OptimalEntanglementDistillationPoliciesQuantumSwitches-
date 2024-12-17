import configparser
import ast
from switchUtils import *
import numpy as np
import random
import math
from itertools import product
import os
import json


class Switch:
    
    
    def __init__(self,switch_config_file,switch_name, fid_thresh, distillFlag, x_value = 0):
        
        self.fid_thresh = fid_thresh
        self.configFile = switch_config_file
        self.name = switch_name
        self.distillFlag = distillFlag
        self.possible_actions_dict = {}
        self.x_value = x_value

        self.stateActionCountDict = {}
        self.iteration = 0
        
        config = configparser.ConfigParser()
        config.read(self.configFile)
        
        # define the queue_length.
        # Here queue length is considered property of switch and not users and hence all users connected to the switch have same queue length.
        self.queue_length = config.getint(self.name, 'QUEUE_LENGTH')
        
        self.no_users = config.getint(self.name, 'NO_USERS')
        self.user_pairs = ast.literal_eval(config.get(self.name, 'USER_CONN_TUPLE_LIST'))
        
        if self.distillFlag == True:
            self.possible_actions_dict_filename = f"dict_distillAllowed_noOfUsers_{self.no_users}_fidThresh_{self.fid_thresh}"
            
            if os.path.exists(self.possible_actions_dict_filename):
                pass
            else:
                self.possibleActionsDistill()
                
            with open(self.possible_actions_dict_filename, "r") as file:
                self.possible_actions_dict_load = json.load(file)
            
            for key, value in self.possible_actions_dict_load.items():
                self.possible_actions_dict[int(key)] = np.array(value)
                
                 
        else:
            self.possible_actions_dict_filename = f"dict_onlySwap_noOfUsers_{self.no_users}_fidThresh_{self.fid_thresh}"
            
            if os.path.exists(self.possible_actions_dict_filename):
                pass
            else:
                self.possibleActionsNoDistill()
            
            with open(self.possible_actions_dict_filename, "r") as file:
                self.possible_actions_dict_load = json.load(file)
            
            for key, value in self.possible_actions_dict_load.items():
                self.possible_actions_dict[int(key)] = np.array(value)
        
        #print(self.possible_actions_dict)
        self.users = []#np.empty(self.no_users, dtype = "object")

        # Using a set comprehension to collect unique strings
        self.all_users = {string_item for tuple_item in self.user_pairs for string_item in tuple_item}

        # Converting the set to a list
        self.all_users = list(self.all_users)
        self.all_users = sorted(self.all_users, key=custom_sort_key)
        
        if len(self.all_users) != self.no_users:
            raise ValueError("number of elements in all_users not equal to total number of users")
            
        
        for i in range(self.no_users):
            self.users.append(Users("users_config.ini", self.all_users[i], self.queue_length))

    
    def getR_intrinsic(self, action):
        action = ''.join(map(str, action))
        switchState = self.getSwitchState()
        novel_state = novel_feature(switchState, self.fid_thresh)
        if novel_state not in self.stateActionCountDict:
            self.stateActionCountDict[novel_state] = {}
            self.stateActionCountDict[novel_state][action] = 1
        else:
            if action not in self.stateActionCountDict[novel_state]:
                self.stateActionCountDict[novel_state][action] = 1
            else:
                self.stateActionCountDict[novel_state][action] += 1
        
        r_intrinsic = math.sqrt((1 / (0.01 * math.log(self.iteration + 1) + 1)) / (self.stateActionCountDict[novel_state][action] + 0.01))
        
        return r_intrinsic

        
        
    def switchReset(self):
        self.users = []
        for i in range(self.no_users):
            self.users.append(Users("users_config.ini", self.all_users[i],  self.queue_length))
        
            
    def getSwitchState(self):

        self.switchState = np.zeros((self.no_users, self.queue_length), dtype=np.float32) 
        
        for index, user in enumerate(self.users):
            self.switchState[index] = user.getUserState()
            
        return self.switchState
    
    def takeAction(self, action):
        self.iteration += 1
        #print("Current Obs:",self.getSwitchState())
        #print("Action:", action)
        if any(np.array_equal(action, arr) for arr in self.allActions()) == False:
            raise ValueError("Invalid Action:",action)
        action = action.reshape(self.no_users, self.queue_length)
        
        for user_ind, user_action in enumerate(action):
            user_id = user_ind+1
            non_zero_index = np.nonzero(user_action)[0]
            are_all_zero = np.all(user_action == 0)
            if are_all_zero:
                reward = self.dn(self.users[user_ind], action)
                self.users[user_ind].lle_generator()
                continue
            
            else:
                
                if len(non_zero_index) == 2:
                    
                    reward = self.distill(self.users[user_ind], non_zero_index, action)
                    continue
                    
                elif len(non_zero_index) == 1:
                    #here we ARE SWAPPING, TO PREVENT REPEATED SWAP, ONLY CHECK FORWARD
                    
                    non_zero_value = user_action[non_zero_index[0]]
                    
                    if non_zero_value > user_id:
                        user2_ind = non_zero_value - 1 # for multiple users here we can instead equate user2_ind to value from user_pairs
                        user2Value_ind = np.where(action[user2_ind] == user_id)
                        
                        reward = self.swap(self.users[user_ind], non_zero_index[0], self.users[user2_ind], user2Value_ind, action)
                    elif non_zero_value < user_id:
                        continue
                
            
            self.users[user_ind].lle_generator()
        #print("Reward:", reward)
        #print("Next Switch State:", self.getSwitchState())
                    
        return self.getSwitchState(), reward
                    
        
            
        
    
    def dn(self, user, action):
        user.timeProgression()
        r_intrinsic = self.getR_intrinsic(action)
        return 0 + r_intrinsic # doing nothing has 0 reward
    
    def swap(self, user1, user1_ind, user2, user2_ind, action):
        f1 = user1.popNdArray(user1_ind)
        f2 = user2.popNdArray(user2_ind)
        
        final_fid = ES_fidelity(f1, f2) 
        user1.timeProgression()
        user2.timeProgression()
        r_intrinsic = self.getR_intrinsic(action)
        
        reward = fid_cap_reward_stepFunction(final_fid, self.fid_thresh)
        
        return reward + r_intrinsic
        
    
    def distill(self,user, user_indices, action):
        f1 = user.popNdArray(user_indices[0])
        f2 = user.popNdArray(user_indices[1])
        f12 ,psucc12 = distillation_psucc_fidelity(f1, f2)
        
        user.timeProgression()
        
        if random.random() < psucc12:
            user.appendNdArray(f12)
            user.sortUserState()
        else:
            pass
        
        r_intrinsic = self.getR_intrinsic(action)
        
        #1st option
        #if self.x_value is not None:
            #return f12*s_curve(5 , self.x_value)
        
        #1st option: remove reward
        #2nd option: try response function related to number of swaps 
        #else:
            #return 0
        return 0 + r_intrinsic
        
                   
    '''
    write code as universal action set for current switch config using decorators
    '''
    def possibleActionsDistill(self):
        possible_actions_dict = {}
        possible_actions_array = self.allActions(allPossible = True)
        
        for index, action in enumerate(possible_actions_array):
            possible_actions_dict[index] = action.tolist()
        
        with open(self.possible_actions_dict_filename, "w") as file:
            json.dump(possible_actions_dict, file)
    
    def possibleActionsNoDistill(self):
        possible_actions_dict = {}
        possible_actions_array = self.allActions_onlySwap(allPossible = True)
        
        for index, action in enumerate(possible_actions_array):
            possible_actions_dict[index] = action.tolist()
            
        with open(self.possible_actions_dict_filename, "w") as file:
            json.dump(possible_actions_dict, file)
        
    
    def action_mask(self):
        
        mask = [False]*len(self.possible_actions_dict)
        
        # Generate all actions possible for the current switch state depending upon whether distillation is allowed or not.
        if self.distillFlag == True:
            actions_array = self.allActions()
        else:
            actions_array = self.allActions_onlySwap()
            
        
        # Create action masks for discrete action space
        for key, value in self.possible_actions_dict.items():
            key = int(key)
            is_present = any(np.array_equal(value, subarray) for subarray in actions_array)
            if is_present:
                mask[key] = True
            else:
                mask[key] = False
                
        return mask  
        
    
    def allActions(self, allPossible = False):
        
        if allPossible == True:
            SwitchState = np.array([np.ones(self.queue_length) for _ in range(self.no_users)])
        
        else:
            SwitchState = self.getSwitchState()
        
        # Get the shape of the input array
        shape = SwitchState.shape
        
        # Create a list to store all possible arrays
        possible_arrays = []
    
        # Generate all unique combinations of integers from 1 to len(SwitchState)
        integer_combinations = set(product(range(0, len(SwitchState) + 1), repeat=np.prod(shape)))
    
        for combination in integer_combinations:
            # Create a new array with the same shape as SwitchState
            new_array = np.array(combination).reshape(shape)
            
            # Apply the condition: If SwitchState element is 0, set the corresponding element to 0
            new_array = np.where(SwitchState == 0, 0, new_array)
            
            # Append the new array to the list of possible arrays
            for i in range(len(new_array)):
                all_conds = False
                if np.count_nonzero(new_array[i]) <= 2:
                    target_integer = i+1
                    exclude_integers = [target_integer, 0]
                    boolean_array_distill = (new_array[i] == target_integer)
                    not_in_exclude = ~np.isin(new_array[i], exclude_integers)
                    
                    count_distill = np.count_nonzero(boolean_array_distill)
                    count_ES = np.count_nonzero(not_in_exclude)
                    if count_distill == 1:
                        all_conds = False
                        break
                    elif count_distill == 2:
                        all_conds = True
                    elif count_distill == 0:
                        if count_ES != 1:
                            if np.count_nonzero(new_array[i] == 0) == len(new_array[i]):
                                all_conds = True
                            else:   
                                all_conds = False
                                break
                        elif count_ES == 1:
                            arrays_except_i = [arr for idx, arr in enumerate(new_array) if idx != i]
    
                            # Use np.concatenate to combine the arrays into a single ndarray
                            result = np.concatenate(arrays_except_i)
                            boolean_array_swap_other = (result == target_integer)
                            count_ES_other = np.count_nonzero(boolean_array_swap_other)
                            if count_ES_other == 1:
                                all_conds = True
                            else:
                                all_conds = False
                                break
                        elif count_ES == 0:
                            all_conds = True
                        
                else:
                    all_conds = False
                    break
            if all_conds:
                possible_arrays.append(np.array([item for sublist in new_array for item in sublist]))
                
        unique_sublists = set()

        # Create a new list without duplicate sublists using a list comprehension
        all_actions = [sublist for sublist in possible_arrays if tuple(sublist) not in unique_sublists and not unique_sublists.add(tuple(sublist))]

        return all_actions
    
    
    
    def allActions_onlySwap(self, allPossible = False):
        
        if allPossible == True:
            SwitchState = np.array([np.ones(self.queue_length) for _ in range(self.no_users)])
        
        else:
            SwitchState = self.getSwitchState()
        
        # Get the shape of the input array
        shape = SwitchState.shape
        
        # Create a list to store all possible arrays
        possible_arrays = []
    
        # Generate all unique combinations of integers from 1 to len(SwitchState)
        integer_combinations = set(product(range(0, len(SwitchState) + 1), repeat=np.prod(shape)))
    
        for combination in integer_combinations:
            # Create a new array with the same shape as SwitchState
            new_array = np.array(combination).reshape(shape)
            
            # Apply the condition: If SwitchState element is 0, set the corresponding element to 0
            new_array = np.where(SwitchState == 0, 0, new_array)
            
            # Append the new array to the list of possible arrays
            for i in range(len(new_array)):
                all_conds = False
                if np.count_nonzero(new_array[i]) <= 1:
                    target_integer = i+1
                    exclude_integers = [target_integer, 0]
                    not_in_exclude = ~np.isin(new_array[i], exclude_integers)
                    
                    count_ES = np.count_nonzero(not_in_exclude)
                    if count_ES != 1:
                        if np.count_nonzero(new_array[i] == 0) == len(new_array[i]):
                            all_conds = True
                        else:   
                            all_conds = False
                            break
                    elif count_ES == 1:
                        arrays_except_i = [arr for idx, arr in enumerate(new_array) if idx != i]
    
                        # Use np.concatenate to combine the arrays into a single ndarray
                        result = np.concatenate(arrays_except_i)
                        boolean_array_swap_other = (result == target_integer)
                        count_ES_other = np.count_nonzero(boolean_array_swap_other)
                        if count_ES_other == 1:
                            all_conds = True
                        else:
                            all_conds = False
                            break
                    elif count_ES == 0:
                        all_conds = True
                        
                else:
                    all_conds = False
                    break
            if all_conds:
                possible_arrays.append(np.array([item for sublist in new_array for item in sublist], dtype=np.int8))
                
        unique_sublists = set()

        # Create a new list without duplicate sublists using a list comprehension
        all_actions_only_swap = [sublist for sublist in possible_arrays if tuple(sublist) not in unique_sublists and not unique_sublists.add(tuple(sublist))]

        return all_actions_only_swap
    

    
class Users:
    def __init__(self, user_config_file, user_name, queue_length):
        
        
        self.configFile = user_config_file
        self.name = user_name
        
        
        
        config = configparser.ConfigParser()
        config.read(self.configFile)
        
        self.queue_length = queue_length #config.getint(self.name, 'queue_length')
        self.userState = np.zeros(self.queue_length, dtype=np.float32)
        self.m_star = config.getint(self.name, 'm_star')
        self.f_star = config.getfloat(self.name, 'f_star')
        self.lle_generation_prob = config.getfloat(self.name, 'lle_generation_prob')
        self.es_prob = config.getfloat(self.name, 'es_prob')
        
        self.alpha = (-math.log(self.f_star))/(self.m_star)
        
    def sortUserState(self):
        userState = self.getUserState()
        userState[::-1].sort()
    
    def popNdArray(self, index):
        
        ndArray = self.getUserState()
        # Check if the array is empty; if so, return None
        if ndArray.size == 0:
            return None

        # Get the original value
        value = ndArray[index]

        # Replace the value with 0
        ndArray[index] = 0

        # Return the original last value
        return value

    def appendNdArray(self, value_to_append):
        
        ndArray = self.getUserState()
        # Check if the array is empty or the value to append is not numeric
        if ndArray.size == 0:
            return

        # Shift elements to the right by one position
        ndArray[1:] = ndArray[:-1]

        # Add the new value to the left
        ndArray[0] = value_to_append
    
    def lle_generator(self):
        
        linkProb = random.random()
        if linkProb <= self.lle_generation_prob:
            self.appendNdArray(1)
        
    
    def timeProgression(self):
        
        # Step 1: Subtract alpha from all values in the userState
        self.userState -= self.alpha

        # Step 2: Sort the userState in descending order
        self.sortUserState()

        # Step 3: Replace elements less than "f_star" with zero
        self.userState[self.userState < self.f_star] = 0
        
    
    def getUserState(self):
        return self.userState
        
    