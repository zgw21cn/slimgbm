# -*- coding:utf-8 -*-

import copy_reg
import types
from multiprocessing import Pool
from functools import partial
import pandas as pd
import numpy as np


# use copy_reg to make the instance method picklable,
# because multiprocessing must pickle things to sling them among process
from slimgbm.slimgbm.histogram import Histogram


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class TreeNode(object):
    def __init__(self,is_leaf=False,leaf_score=None,feature=None,threshold=None,left_child=None,right_child=None):
        """
        :param is_leaf: if True, only need to initialize leaf_score. other params are for intermediate tree node
        :param leaf_score: prediction score of the leaf node
        :param feature: split feature of the intermediate node
        :param threshold: split threshold of the intermediate node
        :param left_child: left child node
        :param right_child: right child node
        :param nan_direction: if 0, those NAN sample goes to left child, if 1 goes to right child.
                              goes to left child by default
        """
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_score = leaf_score


class Tree(object):
    def __init__(self):
        self.root = None
        self.min_sample_split = None
        self.colsample_bylevel = None
        self.reg_lambda = None
        self.gamma = None
        self.num_thread = None
        self.max_bins=25
        self.min_child_weight = None
        self.feature_importance = {}

    def calculate_leaf_score(self, Y):
        """
        According to xgboost, the leaf score is :
            - G / (H+lambda)
        """
        return - Y.grad.sum()/(Y.hess.sum()+self.reg_lambda)

    def find_best_threshold(self, data, col):
        """
        :param data:
               the columns of the data: col, 'label', 'grad', 'hess'

        find best threshold for the given feature: col
        """
        best_threshold = None
        best_gain = - np.inf

        if len(np.unique(data[col]))<2:
            return col, best_threshold, best_gain

        selected_data = data[[col, 'label', 'grad', 'hess']]

        hist=Histogram()
        hist.construct_bins(data,col,self.max_bins)
        best_threshold, best_gain=hist.find_best_split(self.reg_lambda,self.gamma)

        return col, best_threshold, best_gain

    def find_best_feature_threshold(self, X, Y):
        """
        find the (feature,threshold) with the largest gain
        if there are NAN in the feature, find its best direction to go
        """
        nan_direction = 0
        best_gain = - np.inf
        best_feature, best_threshold = None, None
        rsts = None

        # for each feature, find its best_threshold and best_gain, finally select the largest gain
        # implement in parallel
        cols = list(X.columns)
        data = pd.concat([X, Y], axis=1)

        func = partial(self.find_best_threshold, data)
        if self.num_thread == -1:
            pool = Pool()
            rsts = pool.map(func, cols)
            pool.close()

        else:
            pool = Pool(self.num_thread)
            rsts = pool.map(func, cols)
            pool.close()

        for rst in rsts:
            if rst[2] > best_gain:
                best_gain = rst[2]
                best_threshold = rst[1]
                best_feature = rst[0]

        return best_feature, best_threshold, best_gain

    def split_dataset(self, X, Y, feature, threshold):
        """
        split the dataset according to (feature,threshold), nan_direction
            if faeture_value < feature_threshold, samples go to left child
            if faeture_value >= feature_threshold, samples go to right child
            if feature_value==NAN and nan_direction==0, samples go to left child.
            if feature_value==NAN and nan_direction==1, samples go to right child.
        """
        X_cols, Y_cols = list(X.columns), list(Y.columns)
        data = pd.concat([X, Y], axis=1)
        right_data, left_data = None, None
        mask = data[feature] >= threshold
        right_data = data[mask]
        left_data = data[~mask]

        return left_data[X_cols], left_data[Y_cols], right_data[X_cols], right_data[Y_cols]

    def build(self, X, Y, max_depth,best_feature=None,best_threshold=None):
        # check if min_sample_split or max_depth or min_child_weight satisfied
        if X.shape[0] < self.min_sample_split or X.shape[0] < self.max_bins or max_depth == 0 or Y.hess.sum() < self.min_child_weight:
            is_leaf = True
            leaf_score = self.calculate_leaf_score(Y)
            return TreeNode(is_leaf=is_leaf, leaf_score=leaf_score)

        #first level
        if max_depth>0 and best_feature is None and best_threshold is None:
            # column sample before splitting each tree node
            X_selected = X.sample(frac=self.colsample_bylevel, axis=1)
            # find the best feature(among the selected features) and its threshold to split
            best_feature, best_threshold, best_gain = self.find_best_feature_threshold(X_selected, Y)

            # if the gain is negative, it means the loss increase after splitting, so we stop split the tree node
            # node that xgboost does not stop, but adopt post pruning instead
            if best_gain < 0:
                is_leaf = True
                leaf_score = self.calculate_leaf_score(Y)
                return TreeNode(is_leaf=is_leaf, leaf_score=leaf_score)

        # if the gain is not negative, we split the data(original X) according to (best_feature,best_threshold) and nan_direction
        # then feed left data to left child, right data to right child
        # build the tree recursively
        left_X, left_Y, right_X, right_Y = self.split_dataset(X,Y,best_feature,best_threshold)

        left_X_selected=left_X.sample(frac=self.colsample_bylevel, axis=1)
        left_best_feature, left_best_threshold, left_best_gain = self.find_best_feature_threshold(left_X_selected, left_Y)

        right_X_selected=right_X.sample(frac=self.colsample_bylevel, axis=1)
        right_best_feature, right_best_threshold, right_best_gain = self.find_best_feature_threshold(right_X_selected, right_Y)

        if left_best_gain>=right_best_gain:
            left_tree = self.build(left_X, left_Y, max_depth-1,left_best_feature,left_best_threshold)
            right_tree = self.build(right_X, right_Y, max_depth=0)
        else:
            left_tree = self.build(left_X, left_Y, max_depth=0)
            right_tree = self.build(right_X, right_Y, max_depth-1,right_best_feature,right_best_threshold)

        # update the feature importance
        if self.feature_importance.has_key(best_feature):
            self.feature_importance[best_feature] += 1
        else:
            self.feature_importance[best_feature] = 0

        return TreeNode(is_leaf=False, leaf_score=None, feature=best_feature, threshold=best_threshold,
                        left_child=left_tree, right_child=right_tree)

    def fit(self, X, Y, max_depth=6, min_child_weight=1, colsample_bylevel=1.0, min_sample_split=10, reg_lambda=1.0, gamma=0.0, num_thread=-1,max_bins=25):
        self.colsample_bylevel = colsample_bylevel
        self.min_sample_split = min_sample_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.num_thread = num_thread
        self.max_bins=max_bins
        self.min_child_weight = min_child_weight
        # build the tree by a recursive way
        self.root = self.build(X, Y, max_depth)

    def _predict(self, treenode, X):
        """
        predict a single sample
        note that X is a tupe(index,pandas.core.series.Series) from df.iterrows()
        """
        if treenode.is_leaf:
            return treenode.leaf_score
        elif pd.isnull(X[1][treenode.feature]):
            if treenode.nan_direction == 0:
                return self._predict(treenode.left_child, X)
            else:
                return self._predict(treenode.right_child, X)
        elif X[1][treenode.feature] < treenode.threshold:
            return self._predict(treenode.left_child, X)
        else:
            return self._predict(treenode.right_child, X)

    def predict(self, X):
        """
        predict multi samples
        X is pandas.core.frame.DataFrame
        """
        preds = None
        samples = X.iterrows()

        func = partial(self._predict, self.root)
        if self.num_thread == -1:
            pool = Pool()
            preds = pool.map(func, samples)
            pool.close()
        else:
            pool = Pool()
            preds = pool.map(func, samples)
            pool.close()
        return np.array(preds)