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

sys.path.insert(0, './src')
from load_data import path_generator
from GraphDataset import GraphDataset
from model import Net

ROOT = "/home/h8lee/DSC180B-A11-Project"
CONFIG = 'conf/reg_defs.yml'
TRAIN_PATH = 'train_data'
TEST_PATH = 'test_data'

def main(args, batch_size=None, valid_frac=None, stopper_size=None, n_epochs=100):   

    if 'test' in args:
        device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(ROOT, CONFIG)) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            definitions = yaml.load(file, Loader=yaml.FullLoader)
            
        features = definitions['features']
        spectators = definitions['spectators']
        labels = definitions['labels']

        nfeatures = definitions['nfeatures']
        nlabels = definitions['nlabels']

        train_files = (random.sample(path_generator('signal', eda=False), 20) + 
                           random.sample(path_generator('qcd', eda=False), 20))
        random.shuffle(train_files);

        training_dir_path = os.path.join(ROOT, TRAIN_PATH)
        train_graph_dataset = GraphDataset(training_dir_path, features, labels, spectators, n_events=1000, n_events_merge=1, 
                                 file_names=train_files)

        print(f"Graph datasets are successfully prepared at {training_dir_path}")

        def collate(items): return Batch.from_data_list(sum(items, []))

        torch.manual_seed(0)
        if valid_frac is None:
            valid_frac = 0.20

        if batch_size is None:
            batch_size = 32
        
        full_length = len(train_graph_dataset)
        valid_num = int(valid_frac*full_length)

        train_dataset, valid_dataset = random_split(train_graph_dataset, [full_length-valid_num,valid_num])

        train_loader = DataListLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        train_loader.collate_fn = collate
        valid_loader = DataListLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
        valid_loader.collate_fn = collate

        train_samples = len(train_dataset)
        valid_samples = len(valid_dataset)

        training_lst = []
        valid_lst = []
        best_vloss = float(np.inf)
        net = Net().to(device) # Model initialization
        optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
        valid_pred_loss = float(np.inf)
        stopper = False # Early stopper to prevent overfitting; converts to True in later epoch once validation loss starts increasing

        if stopper_size is None:
            stopper_size = 30

        t = tqdm(range(0, n_epochs))

        for epoch in t:
            if stopper:
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
                modpath = os.path.join(ROOT, 'simplenetwork_best.pt')
                print('New best model saved to:',modpath)
                torch.save(net.state_dict(),modpath)
            
            if (epoch > stopper_size) & (batch_vloss > valid_pred_loss):
                stopper = True
            else:
                valid_pred_loss = batch_vloss
                
            print(f'At epoch {epoch}, training loss: {training_batch_loss} and validation loss: {batch_vloss}')

        print('\n', f'Through model training process, the lowest recorded validation RMSE is {np.sqrt(best_vloss)}, and the lowest recorded empirical RMSE is {np.sqrt(min(training_lst))}', '\n')
        training_rmse = [np.sqrt(tloss) for tloss in training_lst]
        validation_rmse = [np.sqrt(vloss) for vloss in valid_lst]

        # Refresh all .pt files by deleting `processed` dir
        # os.rmdir('./processed');

        # ====================================
        # TESTING STARTS HERE

        return training_rmse, validation_rmse, best_epoch

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)