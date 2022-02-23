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

        print(f"\nGraph datasets are successfully prepared at {training_dir_path}", '\n')

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

        print(f'Train and validation data are prepared... Going into model training now', '\n')

        training_lst = []
        valid_lst = []
        best_vloss = float(np.inf)
        net = Net().to(device) # Model initialization
        optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
        valid_pred_loss = float(np.inf)
        stopper = False # Early stopper to prevent overfitting; converts to True in later epoch once validation loss starts increasing

        if os.path.exists(os.path.join(ROOT, 'simplenetwork_best.pt')):
            model_path = os.path.join(ROOT, 'simplenetwork_best.pt')
            stopper = True

        if stopper_size is None:
            stopper_size = 30

        t = tqdm(range(0, n_epochs))

        for epoch in t:
            if stopper:
                if os.path.exists(model_path):
                    print(f'Using pre-trained optimized NN weights stored in {model_path}', '\n')
                else:
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
            best_epoch = 0
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
        epochs = [x+1 for x in range(len(validation_rmse))]
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        _ = sns.set(context='notebook',
                   rc={'axes.spines.right':False,
                      'axes.spines.top':False},
                   style='white')

        sns.lineplot(x=epoch, y=train_rmse, color='blue', ax=ax, label='Train RMSE loss')
        sns.lineplot(x=epoch, y=valid_rmse, color='orange', ax=ax, label='Validation RMSE loss');
        ax.plot(best_epoch+1, valid_rmse[best_epoch], marker='*', markerSize=12, color='red', label='Best model saved at');

        _ = ax.legend(frameon=True)
        _ = ax.set_xlabel('# of Epoch')
        _ = ax.set_ylabel('RMSE')
        _ = ax.set_title('Jet mass NN-regressor learning curve', fontdict={
            'size':15,
            'weight':'bold'
        })

        ax.figure.savefig('./notebooks/learning_curve.png', format='png');

        # ====================================
        # TESTING STARTS HERE
        print('\n\n', '='*25)
        print('Testing Phase', '\n')
        test_files = random_test_path_generator()
        test_dir_path = os.path.join(ROOT, TEST_PATH)
        test_graph_dataset = GraphDataset(test_dir_path, features, labels, spectators, n_events=1000, n_events_merge=1, 
                                 file_names=test_files)

        print(f"\nGraph test datasets are successfully prepared at {test_dir_path}", '\n')

        test_loader = DataListLoader(test_graph_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
        test_loader.collate_fn = collate
        test_samples = len(test_graph_dataset)

        test_p = tqdm(enumerate(test_loader), total=test_samples/batch_size)
        test_lst = []
        net = Net().to(device)

        # Retrieve the model weights that produced smallest validation loss
        net.load_state_dict(torch.load(model_path));

        print(f'Making jet mass predictions on test set using weighted NN regressor', '\n')
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

        test_masked = np.ma.masked_invalid(test_lst).tolist()
        test_resolution = [x for x in test_masked if x is not None]

        return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)