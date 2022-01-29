import seaborn as sns
import pandas as pd
import numpy as np

def mass_distribution(df):
	'''
	This function returns histogram of the distribution of 
	jet masses for two types of jet -- QCD and signal
	Visualization will be done using Python `seaborn` library

	Parameters:
	df -- Pandas dataframe; dataframe composed of all jet features
	available in .root files. Incorporate jet feautres of both signal and QCD jets
	'''
	MASS = 'fj_genjetmsd'

	# Text description to display on visualization
	text = f'Average mass of Signal jets: {avg_mass_signal:.5}\n\
Average mass of QCD jets: {avg_mass_qcd:.5}'

	# Plot stacked histogram, two jet types differentiated with different hue
	_ = sns.set(context='notebook', rc={'figure.figsize':(14,8)}, 
            style='ticks', palette='pastel')
	ax = sns.histplot(x=MASS, data=df, hue='Type',
		bins=range(0,1250,125), multiple='stack') 

	# Plot configuration
	_ = ax.set_title('Distribution of jet mass by jet type', fontdict={'size':15, 'weight':'bold'})
	_ = ax.set_xlabel('Generator-level soft drop mass')
	_ = ax.text(550, 300000, text)
	_ = ax.set_xticks(range(0, 1250, 125))
	_ = ax.set_ylabel('Counts')
	_ = ax.spines['right'].set_visible(False)
	_ = ax.spines['top'].set_visible(False)

	# Summary table
	summary = df.groupby('Type').aggregate(Avg_jetmass=(MASS, 'mean'), Med_jetmass=(MASS, 'median'))

	return ax, summary