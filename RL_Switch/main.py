from train_model import *
from predict import *
from multiprocessing import freeze_support

if __name__ == "__main__":
    
    freeze_support()
    multiprocessed_train()
    multiprocessed_sim()
