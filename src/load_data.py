import os
import uproot
import numpy as np
import pandas as pd 
import dask.dataframe as dd

# Temporary script for EDA portion of DSC180B project
# Author: Jayden Lee

#------------------------------------------------------------------------

# In our case, our dataset is splitted into two parts based on 
# the type of jet: QCD and Hbb(and other type of Higgs jets)
# Datasets for each type exist in different locations
# so should be aware of that in processing

# Within each type, test sets for them also exist in separate directory

#------------------------------------------------------------------------

def path_generator(t, eda=True):
	'''
	This helper function generates filepaths to random sets of QCD and Hbb data
	Implementation of it is driven from the pool of data we have access to

	Parameters:
	t -- Type of jets we want to randomly generate datasets of
	eda -- If True, only certain number of data will be generated; if False, generate filepaths to all files
	'''
	lst = []

	if upeer(t) == 'QCD':
        main = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train_mass_qcd/\
QCD_HT{low}to{high}_TuneCP5_13TeV-madgraph-pythia8/'
        if eda:
            num_data = 10
        
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
            all_files = os.listdir(fp)
            samples = random.sample(all_files, k=num_data)
            
            lst += samples
    elif upper(t) == 'HBB':
        main = '/home/h8lee/teams/DSC180A_FA21_A00/a11/train_mass_hbb/\
BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part{}_TuneCP5_13TeV-madgraph_pythia8/'
        if eda:
            num_data = 3
            
        parts = [1,2]
        
        for part in parts:
            fp = main.format(part)
            all_files = os.listdir(fp)
            samples = random.sample(all_files, k=num_data)
            
            lst += samples

    return lst

# Consider putting this in separate script as "PREPARE DATA FOR EDA PART1"
def load_data(fps:list) -> dask.dataframe:
	'''
	This function takes filepaths to multiple .root files, access into them
	and return all data in dask dataframe, which is simply a collection of Pandas dataframes

	For example, if there are 10 filepaths in our input,
	this function will generate dask.dataframe composed of 10 Pandas dataframe
	'''
	# Global variables
	EVENT_NAME = 'Events'
	NUMPY = 'np'
	df = pd.DataFrame()

	attrs = [] # Need to retrieve names of all attributes

	for fp in fps:
		f = uproot.open(fp)
		tree = f[EVENT_NAME]

		if bool(attrs):
			branches = tree.branches
			attrs = [branch.name for branch in branches]

		data = tree.arrays(attrs, library=NUMPY) # Dictionary

	return

# Global variables
EVENT = 'Events'

# Load datasets; assuming that the user is currently under home directory
pwd = os.getcwd()

# Sample Hbb training set
path_hbb = 'teams/DSC180A_FA21_A00/a11/train_mass_hbb/\
BulkGravitonToHHTo4Q_MX-600to6000_MH-15to250_part1_TuneCP5_13TeV-madgraph_pythia8/\
nano_mc2017_1-1_Skim.root'

# Sample QCD training set
path_qcd = 'teams/DSC180A_FA21_A00/a11/train_mass_qcd/\
QCD_HT1000to1500_TuneCP5_13TeV-madgraph-pythia8/\
nano_mc2017_1-101_Skim.root'

# Full file path to Hbb and QCD datasets
fp_hbb = os.path.join(pwd, path_hbb)
fp_qcd = os.path.join(pwd, path_qcd)

# Validate filepath
assert (os.path.exists(fp_hbb) & os.path.exists(fp_qcd)), 'Incorrect filepath'

# Now access & read datafiles using `uproot`
f_hbb = uproot.open(fp_hbb)
f_qcd = uproot.open(fp_qcd)

# Get access to tree version of the data
tree_hbb = f_hbb[EVENT]
tree_qcd = f_qcd[EVENT]

# Get names of all attributes
branches = tree_hbb.branches
attrs = [branch.name for branch in branches]

# Trying to figure out which attributes to neglect in training process
# and if I should unite all QCD attributes into single binary `is_QCD` attribute


