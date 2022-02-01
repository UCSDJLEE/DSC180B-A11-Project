import pandas as pd
import seaborn as sns
import numpy as np

def msoftdrop_genjetmsd(df:pd.DataFrame) -> FacetGrid:
	'''
	
	'''
	# Compute correlation of coefficient firsthand
	assert set(df.columns) == {''}
	r2 = df.corr().iloc[0,1]

	print(f'Correlation of determination between `fj_msoftdrop` and `fj_genjetmsd` \
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
	    data=df_mass, col='jet_type', kind='scatter',
	    hue='jet_type',
	)

	# Plot configuration
	_ = scatterplot.set_xlabels('Reconstructed soft drop mass')
	_ = scatterplot.set_ylabels('Generator-level soft drop mass')

	return scatterplot