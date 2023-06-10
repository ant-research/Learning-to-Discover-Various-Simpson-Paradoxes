# -*- coding: utf-8 -*-

import os
import pandas as pd
from simpson_paradox_finder import SimpsonParadoxFinder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

data_path = './Data/'

file = os.path.join(data_path, 'auto_mpg_clean.csv')
data = pd.read_csv(file)
target = 'mpg'
treatment = 'weight'
features = ['displacement']
# features-- 'displacement', 'horsepower', 'weight', 'acceleration'， 'model year', 'origin'， 'cylinders'

# model parameters
params = {'subgroup_dim': 5, 'hidden_layer': 2, 'hidden_dim': 64,
          'learning_rate': 0.001, 'alpha': 1, 'beta': 10, 'batch_size': 64,
          'epoch': 50, 'seed': 22, 'activation': True, 'find_strong_amp': False}

# run model
spf = SimpsonParadoxFinder(data, target, features, treatment=treatment, is_binary_target=False)
ret = spf.get_simpson_pairs(method='simnet', params=params, verbose=False)
print("\nResult of Simpson's Paradox Finder:")
for t in ret:
    print('T={}, Y={}, Z={}, paradox_types={}'.format(*t))

# group_distribution = spf.get_subgroup_distribution()
# print('\nFeature distribution:\n', group_distribution)

# save output
# output = spf.finder.output
