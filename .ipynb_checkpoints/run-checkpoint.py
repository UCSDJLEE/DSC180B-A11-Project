import os
import sys
import pandas as pd
import numpy as np


sys.path.insert(0, 'src')

from load_data import path_generator, load_jet_features, load_num_sv
from mass_distribution import mass_distribution
from sv_mass_distribution import sv_mass_distribution
from jet_type_validation import jet_type_validation


def main(targets):

    if 'test' in targets:
        qcd_eda_sets = path_generator('qcd', eda=True)
        signal_eda_sets = path_generator('signal', eda=True)

        msg1 = f'Function defined to write paths to couple datasets; loaded {len(qcd_eda_sets)} QCD files and {len(signal_eda_sets)} Signal files'
        print(msg1, '\n')

        df_qcd = load_jet_features(qcd_eda_sets)
        df_signal = load_jet_features(signal_eda_sets)

        msg2 = f'In {len(qcd_eda_sets)} QCD files, there are {df_qcd.shape[0]} jet observations while in {len(signal_eda_sets)} Signal files, there are {df_signal.shape[0]} signal jets'
        print(msg2, '\n')

        df_signal_labels, signal_counts = jet_type_validation(df_signal, 'signal')

        msg3 = f'In {len(signal_eda_sets)} Signal files, different type of signal jets are distributed as:'
        print(msg3)
        with pd.option_context('display.max_rows', None,
                              'display.max_columns', None,
                              'display.precision', 3):
            print(signal_counts)
        print('')

        signal_idx = df_signal_labels.index.tolist()
        df_signal = df_signal.filter(items=signal_idx, axis=0)

        df_qcd['Type'] = 'QCD'
        df_signal['Type'] = 'Signal'
        df_qcd_and_signal = pd.concat([df_qcd, df_signal], axis=0)
        hist, summary = mass_distribution(df_qcd_and_signal)

        msg4 = 'Average and median jet mass for each type of jets'
        print(msg4)
        with pd.option_context('display.max_rows', None,
                              'display.max_columns', None,
                              'display.precision', 3):
            print(summary)

        msg5 = 'With descriptive analysis and EDA covered, we\'re left to train our regressor model that estimates jet mass based on its properties.'
        print(msg5, '\n')
        print('Demo run.py complete.')

    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)

