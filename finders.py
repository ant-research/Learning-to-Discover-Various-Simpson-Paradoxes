# -*- coding: utf-8 -*-

import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from utils import mpcc, frobenius_norm
import copy
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
import time


# --------------  Base Finder class  -------------- #

class BaseFinder:
    """
    A base class for Simpson's paradox discovery.
    """
    def __init__(self, df, target, features, treatment, is_binary_target=None, params={}):
        self.df = df
        self.target = target
        self.features = features
        self.treatment = treatment
        self.is_binary_target = is_binary_target
        self.params = params

    @staticmethod
    def get_k(coeff, p_value, p_threshold):
        """
        Get the association strength between two variables
        :param coeff: pcc
        :param p_value: p
        :param p_threshold: p threshold
        :return:
        """
        if pd.isna(p_value) or p_value > p_threshold:
            return 0
        else:
            return coeff

    def paradox_criterion(self, global_score, subgroup_scores, params):
        """
        paradox_type: "aar", "ar", "yap", "amp", logical strength: YAP=>AR, AR=>AMP, AR=>AAR
        please see
        [1] https://plato.stanford.edu/entries/paradox-simpson/
        [2] "Can you Trust the Trend? Discovering Simpson’s Paradoxes in Social Data"
        """
        def sign(k):
            if k > 0:
                return 1
            elif k < 0:
                return -1
            else:
                return 0

        def aar_criterion(global_k, subgroup_k):
            """
            Averaged Association Reversal Paradox (AAR)

             :param global_k: global association strength.
             :param subgroup_k: association strength of each subgroup.
            """
            globel_sign = sign(global_k)
            subgroup_signs = [sign(k) for k in subgroup_k]
            avg_sign = sign(sum(subgroup_signs) / len(subgroup_signs))
            return globel_sign != avg_sign

        def ar_criterion(global_k, subgroup_k):
            """
            Association Reversal
            """
            ar11 = global_k > 0 and all(k <= 0 for k in subgroup_k)
            ar12 = global_k >= 0 and all(k < 0 for k in subgroup_k)
            ar21 = global_k < 0 and all(k >= 0 for k in subgroup_k)
            ar22 = global_k <= 0 and all(k > 0 for k in subgroup_k)
            return ar11 or ar12 or ar21 or ar22

        def yap_criterion(global_k, subgroup_k):
            """
            Yule’s Association Paradox (YAP)
            """
            return global_k != 0 and all(k == 0 for k in subgroup_k)

        def amp_criterion(global_k, subgroup_k):
            """
            Amalgamation Paradox (AMP)
            """
            return global_k > max(subgroup_k) or global_k < min(subgroup_k)

        # paradox type
        type2func = {
            'AAR': aar_criterion,
            'AR': ar_criterion,
            'YAP': yap_criterion,
            'AMP': amp_criterion
        }
        
        p_threshold = params.get('p_alpha', 0.05)
        paradox_type = params.get('paradox_type', None)

        # the number of subgroups must be greater than 1
        if len(subgroup_scores) < 2:
            return []

        global_k = self.get_k(*global_score, p_threshold)
        subgroup_k = [self.get_k(*ss, p_threshold) for ss in subgroup_scores]

        if paradox_type is None:
            # return the most strict paradox type
            for pt in ['YAP', 'AR']:
                func = type2func[pt]
                if func(global_k, subgroup_k):
                    return [pt]
            ret = []
            for pt in ['AMP', 'AAR']:
                func = type2func[pt]
                if func(global_k, subgroup_k):
                    ret.append(pt)
            return ret
        else:
            func = type2func[paradox_type]
            if func(global_k, subgroup_k):
                return [paradox_type]
            else:
                return []


# --------------  NaiveFinder  -------------- #

class NaiveFinder(BaseFinder):
    """
    An implementation of the method from paper "Detecting Simpson’s Paradox" (FLAIRS 2018)
    Treatment and Target are continuous variables.
    Features must be categorical variables (discrete variables with a few distinct values)
    """
    _name_ = 'naive'

    def __init__(self, df, target, features, treatment, is_binary_target, params={}):
        super(NaiveFinder, self).__init__(df, target, features, treatment, is_binary_target, params)

        # distinct values of each feature
        self.var2values = {}
        for condition_var in self.features:
            self.var2values[condition_var] = self.df[condition_var].unique()

        self._global_scores = []

    def run(self, verbose=False):
        ret = []  # save paradox results

        for paradox_var in self.treatment:
            # paradox_var is treatment
            y = self.df[self.target]
            x = self.df[paradox_var]
            coeff, pvalue = pearsonr(y.to_numpy(), x.to_numpy())  # calculate pearson correlation coefficient
            global_score = (coeff, pvalue)
            self._global_scores.append(global_score)
            print(f'\n[Global] Treatment={paradox_var}, Target={self.target}, pcc={coeff:.4f}, p-value={pvalue:.4f}, size={len(y)}\n')

            subgroup_scores = []
            for condition_var in self.features:  # loop for each conditioning variable
                if paradox_var == condition_var:
                    continue

                if len(self.var2values[condition_var]) > 50:
                    print(f"This variable with {len(self.var2values[condition_var])} distinct values may be not suitable for a condition variable!")
                    continue

                for v in self.var2values[condition_var]:
                    # loop for each discrete value of the conditioning variable
                    print(f'[Subgroup] Condition on {condition_var}={v}', end=' ')

                    subgroup = self.df[condition_var] == v
                    y = self.df[subgroup][self.target]
                    x = self.df[subgroup][paradox_var]
                    subgroup_size = len(y)

                    try:
                        coeff_condition, pvalue_condition = pearsonr(y.to_numpy(), x.to_numpy())
                    except ArithmeticError:
                        if verbose:
                            print('Treatment or target is constant in this subgroup, or the size is less than 2 . Skip!')
                        coeff_condition = np.nan
                        pvalue_condition = np.nan

                    subgroup_scores.append((coeff_condition, pvalue_condition))
                    print(f'pcc={coeff_condition:.4f}, p-value={pvalue_condition:.4f}, size={subgroup_size}')

                paradox_types = self.paradox_criterion(global_score, subgroup_scores, self.params)

                # ret record the final result (t, c, score_full, paradox_types)
                ret.append([paradox_var, self.target, condition_var, ','.join(paradox_types)])

        return ret


# --------------  TrendFinder  -------------- #

class TrendFinder(BaseFinder):
    """
    An implementation of the "Trend Simpson's Paradox Algorithm" from paper
    "Can you Trust the Trend?: Discovering Simpson's Paradoxes in Social Data" (WSDM 2018)
    """
    _name_ = 'trend'

    def __init__(self, df, target, features, treatment, is_binary_target, params={}):
        super(TrendFinder, self).__init__(df, target, features, treatment, is_binary_target, params)
        self.model = sm.Logit if is_binary_target else sm.OLS

        # distinct values of each feature
        self.var2values = {}
        for condition_var in self.features:
            self.var2values[condition_var] = self.df[condition_var].unique()

        self._global_scores = []

    def run(self, verbose=False):
        ret = []

        for paradox_var in self.treatment:
            # paradox_var is treatment 
            y = self.df[[self.target]]
            x = self.df[[paradox_var]]
            x = sm.add_constant(x)
            model = self.model(y, x).fit(disp=verbose)
            coeff_paradox = model.params[1]
            pvalue_paradox = model.pvalues[1]
            global_score = (coeff_paradox, pvalue_paradox)
            self._global_scores.append(global_score)
            if verbose:
                print(f'\n[Global] Treatment {paradox_var}, NULL model coeff={coeff_paradox:.4f}, p-value={pvalue_paradox:.4f}, size={len(y)}\n')

            subgroup_scores = []
            for condition_var in self.features:   # loop for each conditioning variable
                if paradox_var == condition_var:
                    continue

                for v in self.var2values[condition_var]:
                    # loop for each discrete value of the conditioning variable
                    if verbose:
                        print(f'[Subgroup] Condition on {condition_var}={v}', end=' ')

                    subgroup = self.df[condition_var] == v
                    y = self.df[subgroup][[self.target]]
                    x = self.df[subgroup][[paradox_var]]
                    subgroup_size = len(y)

                    try:
                        x = sm.add_constant(x, has_constant='raise')
                    except ArithmeticError:
                        if verbose:
                            print('treatment or target is constant in this subgroup. skip!')
                        coeff_condition = np.nan
                        pvalue_condition = np.nan
                    else:
                        try:
                            model = self.model(y, x).fit(disp=verbose)
                        except ArithmeticError:
                            print('Encounter exception')
                            coeff_condition = np.nan
                            pvalue_condition = np.nan
                        else:
                            coeff_condition = model.params[1]
                            pvalue_condition = model.pvalues[1]

                    subgroup_scores.append((coeff_condition, pvalue_condition))
                    if verbose:
                        print(f'subgroup score: coeff={coeff_condition:.4f}, p-value={pvalue_condition:.4f}, size={subgroup_size}')

                paradox_types = self.paradox_criterion(global_score, subgroup_scores, self.params)

                # ret record the final result (t, c, score_full, paradox_types)
                ret.append([paradox_var, self.target, condition_var, ','.join(paradox_types)])

        return ret


# -------------- SimNet Finder -------------- #

class SimNet(nn.Module):
    """
    SimNet Model for Simpson's paradox detection
    """
    def __init__(self, x_dim=4, h_dim=16, c_dim=4, l_num=1, activation=False):
        super(SimNet, self).__init__()
        self.x_dim = x_dim    # number of features
        self.c_dim = c_dim    # number of subgroups
        self.L = l_num        # number of hidden layers
        self.activation = activation    # use nonlinear activation or not
        self.FCs = nn.ModuleList([nn.Linear(x_dim, h_dim)])
        for i in range(self.L - 1):
            self.FCs.append(nn.Linear(h_dim, h_dim))  # hidden layer
        self.FCs.append(nn.Linear(h_dim, c_dim))      # subgroup generator

    def forward(self, x):
        for i in range(self.L + 1):
            x = self.FCs[i](x)
            if self.activation:
                x = F.selu(x)
        return F.softmax(x, dim=1)


class SimNetFinder(BaseFinder):
    _name_ = 'simnet'

    def __init__(self, df, target, features, treatment, params={}):
        super(SimNetFinder, self).__init__(df, target, features, treatment,
                                           is_binary_target=None, params=params)
        """
        df: dataframe
        target: str
        features: list of feature columns
        treatment: str or list, if None then loop trough all features
        is_binary_target: needed for naive finder
        """
        self.df['ID'] = [i for i in range(len(self.df))]
        self.global_pcc = {}
        self.model_states = {}
        self.output = self.df
        self.global_scores = {}

        # Hyper-parameters
        self.x_dim = len(self.features)
        self.t_dim = 1  # len(self.treatment)
        self.one_hot_cols = self.params.get('one_hot', [])   # variables to be one-hot encoded
        self.use_t = self.params.get('use_t', False)         # include treatment in SimNet model input
        self.h_dim = self.params.get('hidden_dim', 16)       # dimension of the hidden layer
        self.c_dim = self.params.get('subgroup_dim', 4)      # number of subgroups
        self.layer = self.params.get('hidden_layer', 2)      # number of hidden layers
        self.lr = self.params.get('learning_rate', 0.01)     # learning rate of the CFG
        self.alpha = self.params.get('alpha', 1)             # the sign of mpcc loss
        self.beta = self.params.get('beta', 1)               # the weight of norm loss
        self.num_epochs = self.params.get('epoch', 20)       # the number of training epochs
        self.seed = self.params.get('seed', 22)              # seed number
        self.batch_size = self.params.get('batch_size', 32)  # batch size of training data
        self.activation = self.params.get('activation', False)   # use nonlinear activation or not
        self.find_strong_amp = self.params.get('find_strong_amp', False)  # find strong amp or not
        self.device = self.params.get('device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))  # Device configuration

    def get_data(self, df, t):
        pcc, p = pearsonr(df[t].to_numpy(), df[self.target].to_numpy())       # calculate pearson correlation coefficient
        X_df = df[[f for f in self.features if f not in (t, self.target)]]
        one_hot_cols = [c for c in self.one_hot_cols if c in list(X_df)]
        X_df = pd.get_dummies(X_df, columns=one_hot_cols)         # one hot encoding
        fea_col = list(X_df)
        self.x_dim = len(fea_col)
        Y = torch.from_numpy(df[self.target].values)
        T = torch.from_numpy(df[t].values)
        X = torch.from_numpy(X_df.values)
        ID = torch.from_numpy(df['ID'].values)
        data = TensorDataset(X.float(), T.float(), Y.float(), ID)
        return data, pcc, p, fea_col

    def get_subgroup_name(self, treatment_name):
        if len(self.features) < 2:
            return self.features[0] + '_subgroup'
        else:
            return treatment_name + '_latent_subgroup'

    def run(self, verbose=False):
        """run the SimNet model with MPCC objective on given data"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        all_result = []

        for n in range(len(self.treatment)):
            subgroup_name = self.get_subgroup_name(self.treatment[n])

            print('\n>>> Treatment is {:s} ...'.format(self.treatment[n]))

            # Data loader
            train_data, train_pcc, train_p, fea_col = self.get_data(self.df, self.treatment[n])
            train_data_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
            model = SimNet(x_dim=self.x_dim, h_dim=self.h_dim,
                           c_dim=self.c_dim, l_num=self.layer,
                           activation=self.activation).to(self.device)  # create a model
            # model.apply(init_weights)   # initialize model parameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)      # optimizer
            self.model_states[self.treatment[n]] = copy.deepcopy(model.state_dict())  # save initial model
            self.global_pcc[self.treatment[n]] = train_pcc  # save global pcc

            best_loss = 1e8
            if train_pcc < 0:
                self.alpha = -1 * self.alpha

            if self.find_strong_amp:
                self.alpha = -1 * self.alpha

            if verbose:
                print("\n* [SimNet Training Phase] Subgroup generation is started *")

            s_time = time.time()
            for epoch in range(self.num_epochs):
                one_mpcc, one_norm = [], []
                for i, (x, t, y, _) in enumerate(train_data_loader):
                    x = x.to(self.device).view(-1, self.x_dim)
                    c = model(x)

                    if torch.any(torch.isnan(c)) and verbose:
                        print('Epoch[{}/{}]--Step {}, The subgroup is nan!!!'.format(epoch + 1, self.num_epochs, i))

                    group_pcc, _ = mpcc(t.view(-1, 1), y.view(-1, 1), c)  # mpcc and p-value
                    corr_loss = torch.mean(group_pcc)  # correlation loss
                    # mpcc_loss = torch.abs(train_pcc - corr_loss)
                    frob_loss = frobenius_norm(c)      # Frobenius norm loss

                    if torch.any(torch.isnan(corr_loss)) and verbose:
                        print("Epoch[{}/{}], Step {}, Train mpcc: {:.3f}, Train Norm: {:.3f}"
                              .format(epoch + 1, self.num_epochs, i + 1, corr_loss, frob_loss))

                    loss = self.alpha * corr_loss + self.beta * frob_loss       # total loss
                    # loss = -1 * mpcc_loss + self.beta * frob_loss  # total loss (optional)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    one_mpcc.append(corr_loss.item())
                    one_norm.append(frob_loss.item())

                avg_train_mpcc = sum(one_mpcc) / len(one_mpcc)
                avg_train_norm = sum(one_norm) / len(one_norm)
                avg_loss = self.alpha * avg_train_mpcc + avg_train_norm

                # save the best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.model_states[self.treatment[n]] = copy.deepcopy(model.state_dict())

                if verbose:
                    # Output the mean of the results of all batches on the training set
                    print("Epoch[{}/{}]--Train MPCC: {:.3f}, NORM: {:.3f}, Best Loss: {:.3f}"
                          .format(epoch + 1, self.num_epochs, avg_train_mpcc, avg_train_norm, best_loss))

            run_time = time.time() - s_time
            # load the model with the best result to generate the subgroup
            if verbose:
                print("\n* [SimNet Running time] All training time: {:.3f}s, time per epoch: {:.3f}s".format(run_time, run_time/self.num_epochs))
                # print the number of parameters and FLOPs of the SimNet model
                flops = FlopCountAnalysis(model, x)
                print("\nThe size of the last batch is {}\n".format(len(x)))  # 推荐
                print(flop_count_str(flops))    # 推荐
                print("\n* [SimNet Prediction Phase] Subgroup generation by the best model *")

            model.load_state_dict(self.model_states[self.treatment[n]])  # load the best model
            all_col = [self.treatment[n], self.get_subgroup_name(self.treatment[n]), self.target, 'ID']
            res_df = pd.DataFrame(columns=all_col)
            with torch.no_grad():
                one_mpcc, one_norm = [], []
                for i, (x, t, y, idx) in enumerate(train_data_loader):
                    x = x.to(self.device).view(-1, self.x_dim)
                    c = model(x)
                    group_pcc, _ = mpcc(t.view(-1, 1), y.view(-1, 1), c)  # mpcc and p-value
                    corr_loss = torch.mean(group_pcc)  # correlation loss
                    frob_loss = frobenius_norm(c)  # Frobenius norm loss
                    one_mpcc.append(corr_loss.item())
                    one_norm.append(frob_loss.item())
                    c = c.argmax(1)
                    res_cat = torch.cat([  # x.float().view(-1, self.x_dim),
                        t.float().view(-1, self.t_dim),
                        c.float().view(-1, 1),
                        y.float().view(-1, 1),
                        idx.float().view(-1, 1)], dim=1)
                    onebatch_df = pd.DataFrame(res_cat, columns=all_col).astype("float")
                    res_df = pd.concat([res_df, onebatch_df], axis=0, ignore_index=False)
                avg_mpcc = sum(one_mpcc) / len(one_mpcc)
                avg_norm = sum(one_norm) / len(one_norm)

                if verbose:
                    print("Prediction MPCC: {:.3f}, NORM: {:.3f}".format(avg_mpcc, avg_norm))

            # calculate mpcc
            Y = torch.from_numpy(res_df[self.target].values).float()
            T = torch.from_numpy(res_df[self.treatment[n]].values).float()
            C = torch.from_numpy(res_df[subgroup_name].values)
            C = torch.nn.functional.one_hot(C.long()).float()
            group_pcc, group_p = mpcc(T.view(-1, self.t_dim), Y.view(-1, 1), C.view(-1, C.shape[1]))
            score = torch.mean(group_pcc).item()
            print(f'\n[Subgroup] Average MPCC={score:.4f}')

            subgroup_scores = []
            group_df = res_df.groupby(by=self.get_subgroup_name(self.treatment[n]))
            for i, sub_df in group_df:
                if len(sub_df) > 1:
                    one_pcc, one_p = pearsonr(sub_df[self.treatment[n]], sub_df[self.target])
                else:
                    if verbose:
                        print("the size of this subgroup is not larger than 1!")
                    one_pcc, one_p = np.nan, np.nan
                print(f'subgroup {int(i)}: pcc={one_pcc:.4f}, p-value={one_p:.4f}, size={len(sub_df)}')
                subgroup_scores.append((one_pcc, one_p))

            global_score = (train_pcc, train_p)
            self.global_scores[self.treatment[n]] = global_score
            print("\n[Global] PCC={:.4f}, p-value={:.4f}, size={}\n".format(train_pcc, train_p, len(res_df)))

            paradox_types = self.paradox_criterion(global_score, subgroup_scores, self.params)

            # result record the final result (t, c, score_full, paradox_types)
            result = [self.treatment[n], self.target, subgroup_name, ','.join(paradox_types)]
            # print('T={}, C={}, paradox_types={}'.format(*result))

            all_result.append(result)

            # save subgroup results to output
            subgroup_df = res_df[[self.get_subgroup_name(self.treatment[n]), 'ID']].astype(int)
            self.output = pd.merge(self.output, subgroup_df, on='ID')

        return all_result

    def predict(self, treatment, features):
        # load data
        treatment = treatment[0]
        data, _, _, _ = self.get_data(self.df, treatment)
        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
        # load model
        model = SimNet(x_dim=self.x_dim, h_dim=self.h_dim,
                       c_dim=self.c_dim, l_num=self.layer).to(self.device)  # create a model
        model.load_state_dict(self.model_states[treatment])  # load the best model in dataset
        all_col = features.copy()
        all_col.extend([treatment, self.get_subgroup_name(treatment), self.target])
        res_df = pd.DataFrame(columns=all_col)
        # prediction by mode

        with torch.no_grad():
            one_mpcc, one_norm = [], []
            for i, (x, t, y) in enumerate(data_loader):
                x = x.to(self.device).view(-1, self.x_dim)
                c = model(x)
                group_pcc, _ = mpcc(t.view(-1, 1), y.view(-1, 1), c)  # mpcc and p-value
                corr_loss = torch.mean(group_pcc)  # correlation loss
                frob_loss = frobenius_norm(c)  # Frobenius norm loss
                one_mpcc.append(corr_loss.item())
                one_norm.append(frob_loss.item())
                c = c.argmax(1)
                res_cat = torch.cat([x.float().view(-1, self.x_dim),
                                     t.float().view(-1, self.t_dim),
                                     c.float().view(-1, 1),
                                     y.float().view(-1, 1)], dim=1)
                onebatch_df = pd.DataFrame(res_cat, columns=all_col).astype("float")
                res_df = pd.concat([res_df, onebatch_df], axis=0, ignore_index=False)
        return res_df

    @property
    def _global_scores(self):
        return self.global_scores

    @property
    def _models(self):
        return self.model_states
