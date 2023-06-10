# -*- coding: utf-8 -*-

import pandas as pd
from finders import SimNetFinder, NaiveFinder, TrendFinder
from utils import one_hot


class SimpsonParadoxFinder:
    """
    A tool to discover Simpson's Paradox.
    """
    def __init__(self, df, target=None, features=None, treatment=None,
                 features_onehot=None, is_binary_target=False,
                 method='simnet'):
        """
        df: dataframe
        target: str
        features: list of feature columns
        features_onehot: list of feature columns to be onehot encoding
        treatment: str or list, if None then loop through all features
        is_binary_target: needed for naive finder
        method:
            "naive", ref: "Detecting Simpson’s Paradox" (FLAIRS 2018)
            "trend", ref: "Can you Trust the Trend? Discovering Simpson’s Paradoxes in Social Data" (WSDM 2018)
            "simnet", ref: "Learning to discover various Simpson's paradoxes"（KDD 2023)
        """
        self.finder = None
        self.params = None
        self.df = df
        self.df_origin = df
        self.target = target
        self.features = features              # normalize features can be helpful for simnet
        self.features_origin = features
        self.features_onehot = features_onehot
        if treatment is None:
            self.treatment = features
        elif isinstance(treatment, str):
            self.treatment = [treatment]
        else:
            self.treatment = treatment
        self.is_binary_target = is_binary_target
        self.preprocess()
        self.other_output = None

    def preprocess(self):
        all_cols = self.treatment + [self.target] + self.features
        self.df = self.df[all_cols]
        print('\nTreatment: {}'.format(self.treatment))
        print('Target:    {}'.format([self.target]))
        print('Features: {}\n'.format(self.features))
        print('One hot features: {}\n'.format(self.features_onehot))

        # onehot encoding
        if self.features_onehot:
            self.df = one_hot(self.df, self.features_onehot)
            all_cols = list(self.df)
            not_feature_cols = self.treatment + [self.target]
            self.features = sorted(list(set(all_cols).difference(set(not_feature_cols))))

        # discrete variable mapping
        trans_columns = list(self.df.select_dtypes(exclude=['float64', 'int64']))
        trans_columns = [col for col in all_cols if col in trans_columns]
        print(trans_columns)
        mapper = {}
        for col in trans_columns:
            values = sorted(self.df[col].unique())
            mapping = dict((v, k) for k, v in enumerate(values))
            mapper[col] = mapping
            print('Column {}, mapping {}'.format(col, mapper[col]))
        self.df = self.df.replace(mapper)
        
    def get_simpson_pairs(self, method, params={}, verbose=False):
        self.params = params

        if method == 'simnet':
            self.finder = SimNetFinder(
                self.df, self.target, self.features,
                self.treatment, params=params)
            # ret is a list of (t, c, is_paradox_pair, paradox_type)
            # finder._globel_scores returns a list of pcc of each treatment
            # finder._models returns a list models of each treatment
            ret = self.finder.run(verbose)
            
            subgroup_names = [self.finder.get_subgroup_name(t) for t in self.treatment]
            df_subgroups = self.finder.output[subgroup_names].reset_index(drop=True)
            df_origin_with_subgroups = pd.concat([self.df_origin.reset_index(drop=True), df_subgroups], axis=1)
            self.other_output = {'subgroup_result': df_origin_with_subgroups}
            
        elif method == 'trend':
            self.finder = TrendFinder(
                self.df, self.target, self.features,
                self.treatment, self.is_binary_target, params)
            # ret is a list of (t, c, is_paradox_pair, paradox_type)
            # finder._globel_scores returns a list of beta of each treatment in null model
            ret = self.finder.run(verbose)

        elif method == 'naive':
            self.finder = NaiveFinder(
                self.df, self.target, self.features,
                self.treatment, self.is_binary_target, params)
            # ret is a list of (t, c, is_paradox_pair, paradox_type)
            # finder._globel_scores returns a list of beta of each treatment in null model
            ret = self.finder.run(verbose)

        else:
            raise NotImplementedError(f'method {method} is not implemented')
        
        return ret
         
    def get_global_scores(self):
        return self.finder._global_scores
    
    def get_subgroup_discovery(self):
        return self.other_output['subgroup_result']
    
    def get_subgroup_distribution(self):
        
        def get_dummy_columns(data):
            columns = list(data.select_dtypes(exclude=['float64']))            
            if self.target in columns:
                columns.remove(self.target)
            return columns

        if self.finder._name_ != 'simnet':
            raise NotImplementedError('Not implemented for {} finder'.format(self.finder._name_))

        # output subgorup prediction by mpcc method
        df_subgroup_discovery = self.get_subgroup_discovery().reset_index(drop=True)
        
        # output feature distribution of each subgorup subgroup by mpcc method
        all_cols = [self.target] + self.treatment + self.features_origin
        data = self.df_origin[all_cols]
        df_dummies = pd.get_dummies(data, columns=get_dummy_columns(data)).reset_index(drop=True)
        df_out = []
        for t in self.treatment:
            subgroup_name = self.finder.get_subgroup_name(t)
            df_dummies_with_subgroup = pd.concat([df_dummies, df_subgroup_discovery[[subgroup_name]].astype('int')], axis=1)
            df_subgroup_distribution = df_dummies_with_subgroup.groupby([subgroup_name], as_index=False).mean()
            df_subgroup_distribution.replace(
                {subgroup_name: {i: '{}_{}'.format(subgroup_name, i) for i in range(self.finder.c_dim)}},
                inplace=True
                )
            df_subgroup_distribution.rename({subgroup_name: 'subgroup'}, inplace=True)
            df_out.append(df_subgroup_distribution)
        df_out = pd.concat(df_out, axis=0, ignore_index=True)
        
        return df_out
