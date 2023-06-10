# -*- coding: utf-8 -*-

import os
import pandas as pd
from simpson_paradox_finder import SimpsonParadoxFinder
from utils import one_hot

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)

data_path = './Data/'

file = os.path.join(data_path, 'atlanta_salalry_2015.csv')
data = pd.read_csv(file)

# remove 'Native Hawaiian or Other Pacific'
data = data[data['ethnic.origin'].isin(['Black or African American (Not Hispanic or Latino)', 'White (Not Hispanic or Latino)',
                                        'Hispanic or Latino of any race', 'Asian (Not Hispanic or Latino)',
                                        'American Indian or Alaska Native (Not Hispanic or Latino)',
                                        'Two or More Races (Not Hispanic or Latino)'])]

mapper = {'ethnic.origin': {
            'Hispanic or Latino of any race': 0,
            'American Indian or Alaska Native (Not Hispanic or Latino)': 1,
            'Black or African American (Not Hispanic or Latino)': 2,
            'White (Not Hispanic or Latino)': 3,
            'Two or More Races (Not Hispanic or Latino)': 4,
            'Asian (Not Hispanic or Latino)': 5,
            'Native Hawaiian or Other Pacific': 6},
          'organization': {'EXE Executive Offices': 'EXE',
                           'DPW Department of Public Works': 'DPW',
                           'DWM Department of Watershed Management': 'DWM',
                           'DHR Department of Human Reources': 'DHR',
                           'PRC Parks, Recreation, & Cultural Affairs': 'PRC',
                           'APD Atlanta Police Department': 'APD',
                           'AFR Atlanta Fire & Recuse': 'AFR',
                           'DOA Department of Aviation': 'DOA',
                           'PCD Planning & Community Development': 'PCD',
                           'COR Department of Corrections': 'COR',
                           'DIT Department of Information Technology': 'DIT',
                           'DOF Department of Finance': 'DOF',
                           'CCN City Council': 'CCN',
                           'JDA Municipal Court Operations': 'JDA',
                           'PDA Public Defender Administration': 'PDA',
                           'PCD Atlanta Workforce Development Agency': 'AWD',
                           'LAW Law Department': 'LAW',
                           'AUD Audit Administration': 'AUD',
                           'SOL Solicitor Office': 'SOL',
                           'DOP Department of Procurement': 'DOP',
                           'CRB Administration': 'CRB',
                           'ETH Ethics Administration': 'ETH'}}

# setting
target = 'annual.salary'
treatment = 'sex'
features = ['organization', 'ethnic.origin']   # ethnic.origin, organization, age, sex
data = data.replace(mapper)
features_onehot = ['organization', 'ethnic.origin']  # features

# model parameters
params = {'subgroup_dim': 5, 'hidden_layer': 2, 'hidden_dim': 256,
          'learning_rate': 0.001, 'alpha': 1, 'beta': 10, 'batch_size': 64,
          'epoch': 50, 'seed': 22, 'activation': False, 'find_strong_amp': False}

# run model
spf = SimpsonParadoxFinder(data, target, features, treatment, features_onehot=features_onehot, is_binary_target=False)
ret = spf.get_simpson_pairs(method='simnet', params=params, verbose=False)
print("\nResult of Simpson's Paradox Finder:")
for t in ret:
    print('T={}, Y={}, Z={}, paradox_types={}'.format(*t))

# group_distribution = spf.get_subgroup_distribution()
# print('\nFeature distribution:\n', group_distribution)

# save output
# output = spf.finder.output
