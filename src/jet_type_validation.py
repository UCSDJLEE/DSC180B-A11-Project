import pandas as pd 
import numpy as np

def jet_type_validation(df:pd.DataFrame, jet_type:str):
    '''
    This function validates if all jets in the input dataset 
    belong to one and only one jet type.

    Due to its variety, we will take all types of QCD jets in consideration
    but will only consider `H_bb`, `H_cc`, and `H_qq` for Signal jets

    Parameters:
    df -- Dataframe of jet features; need to include jet types
    jet_type -- Type of jets the function will validate
    '''
    IS_QCDb = 'fj_isQCDb'
    IS_QCDothers = 'fj_isQCDothers'
    IS_HBB = 'fj_H_bb'
    IS_HQQ = 'fj_H_qq'

    if jet_type.upper() == 'QCD':
        # Retrieve only the label columns 
        all_attrs = df.columns.tolist()
        start_idx = all_attrs.index(IS_QCDb)
        end_idx = all_attrs.index(IS_QCDothers)+1
        qcd_labels = all_attrs[start_idx:end_idx]

        df_labels = df[qcd_labels]

        # Below print statement must be True if all jets belong to one type
        print(f'Each jet corresponds to exactly one type: {len(df_labels.sum(axis=1).unique()) == 1}')

        counts = (df_labels.sum(axis=0)
            .sort_values(ascending=False)
            .to_frame(name='Count'))

    elif jet_type.upper() == 'SIGNAL':
        # Retrieve only the label columns 
        all_attrs = df.columns.tolist()
        start_idx = all_attrs.index(IS_HBB)
        end_idx = all_attrs.index(IS_HQQ)+1
        signal_labels = all_attrs[start_idx:end_idx]

        df_labels = df[signal_labels]

        # We're only going to include signal jets
        # of types H_bb, H_cc, H_qq for performing EDA
        # since these three types of Higgs jets 
        # are the most common elementary particles
        # Higgs bosons decay into
        df_labels = df_labels[
            (df_labels['fj_H_bb'] == 1) |
            (df_labels['fj_H_cc'] == 1) |
            (df_labels['fj_H_qq'] == 1)
        ]

        # Drop observations that are associated to more than single type
        df_labels['temp'] = df_labels['fj_H_bb'] + df_labels['fj_H_cc'] + df_labels['fj_H_qq']
        print(f'After dropping uninterested jet types: {df_labels.shape[0]} rows', '\n')

        df_labels = df_labels[df_labels['temp'] == 1].drop(columns='temp')
        print(f'After dropping jets with more than one type associated: {df_labels.shape[0]} rows', '\n')

        # Below print statement must be True if all jets belong to one type
        print(f'Each jet corresponds to exactly one type: {len(df_labels.sum(axis=1).unique()) == 1}')

        counts = (df_labels.sum(axis=0)
            .sort_values(ascending=False)
            .to_frame(name='Count'))

    else:
        return

    return df_labels, counts


