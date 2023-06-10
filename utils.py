# -*- coding: utf-8 -*-

import numpy as np
import scipy.special as special
import torch
import pandas as pd


# Calculate the multi-group pearson correlation coefficient
def mpcc(t, y, c):
    """
    Multi-group Pearson Correlation Coefficient (MPCC).
    :param t: Treatment  (B×1)  # B is the batch size.
    :param y: Outcome    (B×1)
    :param c: Confounder (B×C)  # C is the number of groups.
    :return: the pcc and p-value of each group.
    """
    yt_hadamard = y.mul(t)
    yy_hadamard = y.mul(y)
    tt_hadamard = t.mul(t)
    Cy = c.t() @ y
    Ct = c.t() @ t
    sum_c = torch.sum(c, dim=0)
    sum_c = sum_c.view(-1, 1)
    sum_yt = c.t() @ yt_hadamard
    sum_yy = c.t() @ yy_hadamard
    sum_tt = c.t() @ tt_hadamard

    # numerator
    cov_yt = sum_c.mul(sum_yt) - Cy.mul(Ct)

    # denominator
    Dy = sum_c.mul(sum_yy) - Cy.mul(Cy)
    Dt = sum_c.mul(sum_tt) - Ct.mul(Ct)
    Dy = torch.where(Dy <= 0, torch.full_like(Dy, 1e-8), Dy)
    Dt = torch.where(Dt <= 0, torch.full_like(Dt, 1e-8), Dt)

    # take groups with size > 0
    nonzero_group = torch.nonzero(sum_c, as_tuple=True)
    cov_yt = cov_yt[nonzero_group]
    Dy = Dy[nonzero_group]
    Dt = Dt[nonzero_group]

    sigma_yt = torch.sqrt(Dy).mul(torch.sqrt(Dt))

    # mpcc calculation
    group_pcc = torch.zeros_like(sigma_yt)
    nonzero_idx = torch.nonzero(sigma_yt, as_tuple=True)
    group_pcc[nonzero_idx] = cov_yt[nonzero_idx] / sigma_yt[nonzero_idx]
    group_pcc = group_pcc[nonzero_idx]

    # p-value calculation
    with torch.no_grad():
        p_value = []
        for i in range(len(group_pcc)):
            r = group_pcc[i]
            ab = sum_c[i] / 2 - 1
            p = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r)))) if ab != 0 else 1
            p_value.append(p)

    return group_pcc, p_value


def frobenius_norm(c):
    """
    Calculate the frobenius norm loss
    """
    n = c.shape[0]
    k = torch.as_tensor(c.shape[1]).float()
    frob = (torch.sqrt(k) / n) * torch.norm(torch.sum(c, dim=0)) - 1
    return frob


def one_hot(df, columns):
    """
    Onehot encoding function.
    :param df: dataframe
    :param columns: features to be onehot encoded.
    :return: dataframe
    """
    return pd.get_dummies(df, columns=columns)
