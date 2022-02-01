import pandas as pd 
import numpy as np 
import seaborn as sns

def sv_mass_distribution(df):
    '''
    This function returns multiple boxplot of the distribution of
    jet masses separated by number of secondary vertices recorded
    in the jet. For example, if all jets in the dataset have upto 3 secondary vertices,
    this function creates 4 different boxplot(0, 1, 2, 3), displaying distribution of jet masses
    according to count of secondary vertices

    Parameters:
    df -- (n x 2) dataframe; number of secondary vertices(# of SVs recorded) and jet mass(generator-level soft drop mass) must be present
    '''
    assert list(df.columns) == ['# of SVs recorded', 'generator-level soft drop mass']
    XAXIS = 'Number of secondary vertices recorded'
    TITLE = 'Distribution of jet mass per jet\nwith different number of recorded SVs in a jet'
    YAXIS = 'Generator-level soft drop mass'
    font_style = {
        'size':20,
        'weight':'bold',
        'color':'grey'
    }

    # Plot boxplot using `seaborn` APIs
    _ = sns.set(rc={'figure.figsize':(12,8)})
    box = sns.boxplot(x='# of SVs recorded', y='generator-level soft drop mass',
        data=df, palette='flare')

    # Plot configuration
    _ = box.set_title(TITLE, fontdict=font_style)
    _ = box.set_xlabel(XAXIS)
    _ = box.set_ylabel(YAXIS)

    # Summary dataframe
    summary = df.groupby('# of SVs recorded').aggregate(Avg_jetmass=('generator-level soft drop mass', 'mean'), Median_jetmass=('generator-level soft drop mass', 'median')).sort_index()

    return box, summary
