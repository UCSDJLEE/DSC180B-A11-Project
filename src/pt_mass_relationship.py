import pandas as pd 
import numpy as np 
import seaborn as sns

def pt_mass_relationship(df:pd.DataFrame):
    '''
    This function reads the dataframe
    to sketch scatterplot with transverse moment of jets(`fj_pt`) on horizontal axis
    and jet mass, or our target variable(`fj_genjetmsd`) on vertical axis

    Parameters:
    df -- Dataframe consisting of transverse momentum of jets, their mass,
    and their types(QCD or Signal)
    '''
    # Sketch scatterplot
    _ = sns.set(context='notebook', style='ticks',
        palette='pastel', rc={'figure.figsize':(12,8)})
    scatterplot = sns.relplot(kind='scatter', x='fj_pt', 
        y='fj_genjetmsd', data=df, col='Type')

    # Axis labeling
    _ = scatterplot.set_xlabels('Transverse momentum')
    _ = scatterplot.set_ylabels('Generator-level soft drop mass')

    return scatterplot
