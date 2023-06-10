# -*- coding: utf-8 -*-

import os
import pandas as pd
from simpson_paradox_finder import SimpsonParadoxFinder
from sklearn.datasets import load_iris

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = './Data/'

file = os.path.join(data_path, 'UCB_admissions_data.csv')
data = pd.read_csv(file)
target = 'Admit'
treatment = 'Gender'
features = ['Dept']
features_onehot = ['Dept']

# model parameters
params = {'subgroup_dim': 2, 'hidden_layer': 2, 'hidden_dim': 64,
          'learning_rate': 0.01, 'alpha': 1, 'beta': 10, 'batch_size': 64,
          'epoch': 50, 'seed': 22, 'activation': False, 'find_strong_amp': False}

# run model
spf = SimpsonParadoxFinder(data, target, features, treatment=treatment, features_onehot=features_onehot, is_binary_target=False)
ret = spf.get_simpson_pairs(method='simnet', params=params, verbose=False)

print("\nResult of Simpson's Paradox Finder:")
for t in ret:
    print('T={}, Y={}, Z={}, paradox_types={}'.format(*t))

group_distribution = spf.get_subgroup_distribution()
print('\nFeature distribution:\n', group_distribution)

# save output
# output = spf.finder.output
