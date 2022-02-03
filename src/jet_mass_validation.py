import pandas as pd
import seaborn as sns
import numpy as np

def jet_mass_validation(df:pd.DataFrame):
	'''
	This function uses the two mass variables, `fj_msoftdrop`
    and `fj_genjetmsd`, to demonstrate the relationship
    between the two. The result will prove how useful 
    `fj_msoftdrop` can be in training our model
    
    Parameters:
    df -- dataframe consisting two mass variables
    and a label column that indicate the type of jet
	'''
	# Compute correlation of coefficient firsthand
	r2 = df.corr().iloc[0,1]

	print(f'Correlation coefficient between `fj_msoftdrop` and `fj_genjetmsd` \
is {r2:.4}')

	# Linear regression plot

	rc_params = {
	    'figure.figsize':(12,8),
	    'axes.spines.right':False,
	    'axes.spines.top':False
	}

	_ = sns.set(context='notebook', style='white', rc=rc_params, palette='pastel')
	scatterplot = sns.relplot(
	    x='fj_msoftdrop', y='fj_genjetmsd',
	    data=df, col='jet_type', kind='scatter',
	    hue='jet_type',
	)

	# Plot configuration
	_ = scatterplot.set_xlabels('Reconstructed soft drop mass')
	_ = scatterplot.set_ylabels('Generator-level soft drop mass')

	return scatterplot
