import os
import uproot
import numpy as np
import pandas as pd 
import random

# Data loader script for EDA portion of DSC180B project
# Author: Jayden Lee

#------------------------------------------------------------------------ral

# In our case, our dataset is splitted into two parts based on 
# the type of jet: QCD and Hbb(and other type of Higgs jets)
# Datasets for each type exist in different locations
# so should be aware of that in processing

# Within each type, test sets for them also exist in separate directory

#------------------------------------------------------------------------

# Global variables
EVENTS = 'Events'
NUMPY = 'np'

def path_generator(t:str, eda=True) -> list:
    '''
    This helper function generates filepaths to random sets of QCD and Hbb data
    Implementation of it is driven from the pool of data we have access to; location of all data files

    Parameters:
    t -- Type of jets we want to randomly generate datasets of
    eda -- If True, only certain number of data will be generated; if False, generate filepaths to all files
    '''
    qcd = 'QCD'
    signal = 'SIGNAL'
    lst = []

    if eda:
        if upper(t) == qcd:
            main = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train_mass_qcd/\
QCD_HT{low}to{high}_TuneCP5_13TeV-madgraph-pythia8/'
            num_data = 11
            
            bounds = [
                [1000,1500],
                [1500,2000],
                [2000, 'Inf'],
                [500,700],
                [700,1000]
            ]
            
            for bound in bounds:
                low, high = bound
                fp = main.format(low=low, high=high)
                temp = os.listdir(fp)

                # There exists couple hidden .root files under these directories
                # They are inaccessible, so exclude them from list of files to sample from
                all_files = [file for file in temp if not file.startswith('.')]
                samples = random.sample(all_files, k=num_data)
                files = [os.path.join(fp, sample) for sample in samples]
                lst += files

        elif upper(t) == signal:
            main = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train_mass_hbb/\
BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part{}_TuneCP5_13TeV-madgraph_pythia8/'
            num_data = 4
                
            parts = [1,2]
            
            for part in parts:
                fp = main.format(part)
                all_files = os.listdir(fp)
                samples = random.sample(all_files, k=num_data)
                
                files = [os.path.join(fp, sample) for sample in samples]
                lst += files

   #  else:
   #    if upper(t) == qcd:
   #        main = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train_mass_qcd/\
      #       QCD_HT{low}to{high}_TuneCP5_13TeV-madgraph-pythia8/'
            # num_data = 10

    return lst


def load_jet_features(fps:list) -> pd.DataFrame:
    '''
    This function retrieves all jet features and returns them structured as 
    pd.DataFrame. Implemented to filter out irrelevant, or useless, jet features,
    retrieve features in dictionary, and converts them into pd.DataFrame

    Parameters:
    fps -- List of filepaths; if the paths are for QCD dataset,
    function will return the features of QCD jets as pd.DataFrame
    '''
    jet_features = []
    unnecesssary_attrs = [
        'fj_idx',
        'fj_genRes_mass',
        'fj_lsf3'
    ]
    df = pd.DataFrame()
    
    for fp in fps:
        f = uproot.open(fp)
        tree = f[EVENTS]
        
        if df.empty:
            attrs = [branch.name for branch in tree.branches] # All attributes in datasets
            jet_features += list(filter(lambda x:x.startswith('fj'), attrs)) # Only need jet features
            jet_features = [feat for feat in jet_features if feat not in unnecesssary_attrs] # drop sterile attributes
        
        features = tree.arrays(jet_features, library=NUMPY)
        df = pd.concat([df, pd.DataFrame(features)], axis=0)
        
    df = df.reset_index(drop=True)
    
    return df


def load_num_sv(fps):
    '''
    This function counts number of secondary vertices in the jet
    using `sv_pt_log` attribute.
    All jets have `sv_pt_log` in an array of 7 float values
    as they all get zero-padded to have common length of 7
    For example, jet with 5 recorded secondary vertices in it will have
    `sv_pt_log` of [val1, val2, val3, val4, val5, 0, 0]
    `load_num_sv()` filters out all filler entries, or zeros, and 
    counts the actual number of secondary vertices recorded in the jet

    Parameters:
    fps -- List of filepaths; if the paths are for QCD dataset,
    function returns the number of secondary vertices recorded in each jet
    as well as their corresponding mass
    '''
    SV_PT_LOG = 'sv_pt_log'
    FJ_GENJETMSD = 'fj_genjetmsd'
    num_svs = []
    jet_mass = []
    
    for fp in fps:
        f = uproot.open(fp)
        tree = f['Events']
        sv_pt_logs = tree.arrays(SV_PT_LOG, library=NUMPY)[SV_PT_LOG] # 2D array

        # Exclude filler entries
        num_sv = list(map(lambda sublst: len(list(filter(lambda x: x != 0, sublst))), sv_pt_logs))
        num_svs += num_sv
        
        # Retrieve jet masses, or the target values
        masses = tree.arrays(FJ_GENJETMSD, library=NUMPY)[FJ_GENJETMSD].tolist()
        jet_mass += masses
    
    return num_svs, jet_mass
