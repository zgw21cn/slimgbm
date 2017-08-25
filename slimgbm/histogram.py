# coding=utf-8
from pandas.tools.tile import _bins_to_cuts
from slimgbm.slimgbm.bin import Bin
import pandas as pd
import pandas.core.algorithms as algos
import numpy as np


class Histogram(object):
    def __init__(self):
        self.bins = []

    def _qcut(self, x, q, labels=None, retbins=False, precision=3):
        quantiles = np.linspace(0, 1, q + 1)
        bins = algos.quantile(x, quantiles)
        bins = np.unique(bins)
        return _bins_to_cuts(x, bins, labels=labels, retbins=retbins, precision=precision, include_lowest=True)

    def construct_bins(self, data, col, max_bins):
        """
        :param data: the columns of the data: col, 'label', 'grad', 'hess'
        :param max_bins:
        :return:
        """
        if not isinstance(data, pd.core.frame.DataFrame):
            raise TypeError("data should be a pandas.core.series.Series")

        bins_count = min(max_bins, len(data[col].unique()) - 1)

        bins, cut_points = self._qcut(data[col].values, bins_count, retbins=True)

        for i in range(len(cut_points) - 1):
            mask = bins.codes == i
            grad_sum = data.grad[mask].sum()
            hess_sum = data.hess[mask].sum()

            bin = Bin(cut_points[i], cut_points[i + 1], grad_sum, hess_sum)
            self.bins.append(bin)


    def find_best_split(self, reg_lambda, gamma):
        best_gain = float("-inf")
        best_split=None

        for i in range(1, len(self.bins)):
            GL = 0.0
            HL = 0.0
            GR = 0.0
            HR = 0.0
            for j in range(0, i):
                GL = GL + self.bins[j].grad_sum
                HL = HL + self.bins[j].hess_sum

            for j in range(i, len(self.bins)):
                GR = GR + self.bins[j].grad_sum
                HR = HR + self.bins[j].hess_sum

            gain = 0.5 * (GL ** 2 / (HL + reg_lambda) + GR ** 2 / (HR + reg_lambda)
                          - (GL + GR) ** 2 / (HL + HR + reg_lambda)) - gamma

            if gain > best_gain:
                best_split = self.bins[i].upper_bound
                best_gain = gain

        return best_split, best_gain



    




























