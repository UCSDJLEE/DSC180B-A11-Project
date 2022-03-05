# Neural Network Regressor, Jet Mass
# DSC180B WI22 Group1 Project 

# Necessary imports
import torch
import torch_geometric
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d, Flatten, Module
from torch_scatter import scatter_mean
from torch.utils.data import random_split
from torch_geometric.data import DataListLoader, Batch
from tqdm import tqdm
import numpy as np
import os
import sys
import pandas as pd
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt 
import seaborn as sns 

sys.path.insert(0, './src')
from load_data import path_generator, random_test_path_generator
from prepare_dataloader import prepare_dataloader
from GraphDataset import GraphDataset
from learning_curve import learning_curve
from model import Net

ROOT = "/home/h8lee/DSC180B-A11-Project"
CONFIG = 'conf/reg_defs.yml'
TRAIN_PATH = 'train_data'
TEST_PATH = 'test_data'
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
modpath = os.path.join(ROOT, 'simplenetwork_best.pt')

with open(os.path.join(ROOT, CONFIG)) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    definitions = yaml.load(file, Loader=yaml.FullLoader)

features = definitions['features']
spectators = definitions['spectators']
labels = definitions['labels']

nfeatures = definitions['nfeatures']
nlabels = definitions['nlabels']


def main(args, batch_size=32, valid_frac=None, stopper_size=None, n_epochs=100):   

    if 'train' in args:
        train_files = path_generator('both', eda=False)

        training_dir_path = os.path.join(ROOT, TRAIN_PATH)
        if os.path.exists(training_dir_path):
            print(f'Training data are processed and ready to be employed', '\n')
        else:
            print('Generating graph datasets for training...', '\n')
        
        train_graph_dataset = GraphDataset(training_dir_path, features, labels, spectators, n_events=1000, n_events_merge=1, 
                                 file_names=train_files)

        torch.manual_seed(0)
        if valid_frac is None:
            valid_frac = 0.20
        
        full_length = len(train_graph_dataset)
        valid_num = int(valid_frac*full_length)

        train_dataset, valid_dataset = random_split(train_graph_dataset, [full_length-valid_num,valid_num])

        train_loader = prepare_dataloader(train_dataset, batch_size=batch_size)
        valid_loader = prepare_dataloader(valid_dataset, batch_size=batch_size)

        train_samples = len(train_dataset)
        valid_samples = len(valid_dataset)

        print('\n', f'Train and validation data are prepared... Going into model training now', '\n')

        training_lst = []
        valid_lst = []
        best_vloss = float(np.inf)
        net = Net().to(device) # Model initialization
        optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
        valid_pred_loss = float(np.inf) # Tracker for validation loss from previous epoch
        stopper = False # Early stopper to prevent overfitting; converts to True in later epoch once validation loss starts increasing

        if stopper_size is None:
            stopper_size = 30

        t = tqdm(range(0, n_epochs))

        for epoch in t:
            if stopper:
                print('Early stopping enforced')
                break;
            
            p = tqdm(enumerate(train_loader), total=train_samples/batch_size, leave=bool(epoch==n_epochs-1))
            q = tqdm(enumerate(valid_loader), total=valid_samples/batch_size, leave=bool(epoch==n_epochs-1))
            
            loss_func = nn.MSELoss() # Mean Squared Error to track learning 
            if (epoch > 0) & (os.path.exists('simplenetwork_best.pt')):
                net.load_state_dict(torch.load('simplenetwork_best.pt'))

            # Training with training set
            training_temp = []
            valid_temp = []
            net.train()
            for i, data in p:
                data = data.to(device)
                y = data.y
                prediction = net(data.x, data.batch) 
                loss = loss_func(prediction.float(), y.float())
                training_temp.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            
            training_batch_loss = np.average(training_temp)
            training_lst.append(training_batch_loss)
            
            # Evaluating with validation set
            net.eval();
            with torch.no_grad(): 
                for j, vdata in q:
                    vdata = vdata.to(device)
                    y = vdata.y
                    vpreds = net(vdata.x, vdata.batch)
                    vloss = loss_func(vpreds.float(), y.float())
                    
                    valid_temp.append(vloss.item())
                
            batch_vloss = np.average(valid_temp)
            valid_lst.append(batch_vloss)
            
            if batch_vloss < best_vloss:
                best_vloss = batch_vloss
                best_epoch = epoch
                print('\n', f'Best epoch: {best_epoch}')
                print('New best model saved to:',modpath, '\n')
                torch.save(net.state_dict(),modpath)
            
            if (epoch > stopper_size) & (batch_vloss > valid_pred_loss):
                stopper = True
            else:
                valid_pred_loss = batch_vloss
                
            print(f'At epoch {epoch}, training loss: {training_batch_loss} and validation loss: {batch_vloss}')

        if bool(training_lst):
            print('\n', f'Through model training process, the lowest recorded validation RMSE is {np.sqrt(best_vloss)}, and the lowest recorded empirical RMSE is {np.sqrt(min(training_lst))}', '\n')

        training_rmse = [np.sqrt(tloss) for tloss in training_lst]
        validation_rmse = [np.sqrt(vloss) for vloss in valid_lst]
        ax = learning_curve(training_rmse, validation_rmse, best_epoch)

        ax.figure.savefig('./notebooks/learning_curve.png', format='png');

        print('\nModel training complete! To test run the fitted model, run `python3 run.py test` in command line\n')

        # ====================================
        # TESTING STARTS HERE
    elif 'test' in args:
        print('\n\n', '='*25)
        print('Testing Phase...', '\n')
        test_files = random_test_path_generator()
        test_dir_path = os.path.join(ROOT, TEST_PATH)
        test_graph_dataset = GraphDataset(test_dir_path, features, labels, spectators, n_events=1000, n_events_merge=1, 
                                 file_names=test_files)

        print(f"\nGraph test datasets are successfully prepared at {test_dir_path}", '\n')

        test_loader = prepare_dataloader(test_graph_dataset, batch_size=batch_size)
        test_samples = len(test_graph_dataset)

        test_p = tqdm(enumerate(test_loader), total=test_samples/batch_size)
        test_lst = []
        net = Net().to(device)

        # Retrieve the model weights that produced smallest validation loss
        net.load_state_dict(torch.load(modpath));

        print('\n', f'Making jet mass predictions on test set using weighted NN regressor', '\n')
        net.eval();
        with torch.no_grad():
            for k, tdata in test_p:
                tdata = tdata.to(device) # Moving data to memory
                y = tdata.y # Retrieving target variable
                tpreds = net(tdata.x, tdata.batch) 
                loss_t = (tpreds.float() - y.float()) / (y.float())
                loss_t_np = loss_t.cpu().numpy()
                loss = loss_t_np.ravel().tolist()
                test_lst+=loss

        test_masked = np.ma.masked_invalid(test_lst).tolist() # Mask invalid resolution losses
        test_resolution = [x for x in test_masked if x is not None]

        avg_resolution = np.average(test_resolution)
        std_resolution = np.std(test_resolution)

        print(f'Evaluation complete: resolution centered around {round(avg_resolution, 2)}, varying at {round(std_resolution, 2)}...')
        print('\n\n'+'Project demonstration complete.'+'\n\n')
        print('We encourage you to check out our demonstrations under `notebooks` folder in current repo!', '\n')

        return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)