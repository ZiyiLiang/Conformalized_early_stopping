import time
import os
import torch as th
import numpy as np

import pathlib
import sys
from tqdm import tqdm
from copy import deepcopy
from classification import ProbabilityAccumulator as ProbAccum
from scipy.stats.mstats import mquantiles
from sympy import *


class Conformal_PSet:
    '''
    Class for computing marginal or label conditional conformal prediction sets.
    '''
    def __init__(self, net, device, cal_loader, n_classes, model_list, alpha,
                 verbose = True, progress = True, lc=True, random_state = 2023) -> None:
        self.net = net
        self.device = device
        self.cal_loader = cal_loader
        self.n_classes = n_classes
        self.model_list = model_list
        self.alpha = alpha
        self.verbose = verbose
        self.progress = progress
        self.lc = lc       # compute label conditional set or not
        self.random_state = random_state

        if self.verbose:
            print('Calibrating each model in the list...')
            sys.stdout.flush()
        self._calibrate_alpha()
        if self.verbose:
            print('Initialization done!')
            sys.stdout.flush()

    def _calibrate_alpha_single(self, p_hat_cal, cal_labels):
        n_cal = len(cal_labels)
        grey_box = ProbAccum(p_hat_cal)

        rng = np.random.default_rng(self.random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n_cal)
        alpha_max = grey_box.calibrate_scores(cal_labels, epsilon=epsilon)
        scores = self.alpha - alpha_max
        level_adjusted = (1.0-self.alpha)*(1.0+1.0/float(n_cal))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Return the calibrate level
        return self.alpha - alpha_correction

    # Calibrate the alpha level with calibration set for all the models in the list
    # This is one time effort, do not need to repeat this process for new test inputs
    def _calibrate_alpha(self):
        n_model = len(self.model_list)
        alpha_calibrated = np.full((n_model,), -1.0)
        alpha_calibrated_lc = np.full((n_model,self.n_classes), -1.0)

        if self.progress:
            iterator = tqdm(range(n_model))
        else:
            iterator = range(n_model)

        for model_idx in iterator:
            model_path = self.model_list[model_idx]
            self.net.load_state_dict(th.load(model_path))

            p_hat_cal = []
            cal_labels = []

            for inputs, labels in self.cal_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                p_hat_batch = self.net.predict_prob(inputs)
                p_hat_cal.append(p_hat_batch)
                cal_labels.append(labels)

            cal_labels = th.cat(cal_labels)
            p_hat_cal = np.concatenate(p_hat_cal)


            # Use all calibration data to calibrate for the marginal case
            alpha_calibrated[model_idx] = self._calibrate_alpha_single(p_hat_cal, cal_labels)

            if self.lc:
                # Use only calibration data with specified label to calibrate alpha for label conditional case
                for label in range(self.n_classes):
                    idx = cal_labels == label
                    alpha_calibrated_lc[model_idx][label] = self._calibrate_alpha_single(p_hat_cal[idx], cal_labels[idx])

        self.alpha_calibrated = alpha_calibrated
        self.alpha_calibrated_lc = alpha_calibrated_lc

        return None



    def pred_set(self, test_inputs, best_model, marginal=True):
        '''
        Calculate the marginal or label conditional prediction sets
        '''
        n_test = len(test_inputs)
        assert len(best_model) == n_test and len(best_model[0]) == self.n_classes, \
               'Model list should have size of (n_test, n_class), reshape if needed.'

        # if label conditional alpha is not calibrated in initialization stage, cannot compute lc psets
        if not self.lc:
            assert marginal==True, 'Cannot compute label-conditional prediction set, initialized with\
                                                without setting the label-conditional flag on!'

        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        pred_sets= []
        for i in iterator:
            pset = self._pred_set_single(test_inputs[i][None], i, best_model[i], marginal)
            pred_sets.append(pset)
        if True:
            print("Finished computing {} prediction sets for {} test points.".format(['label conditional', 'marginal'][marginal], n_test))
            sys.stdout.flush()
        return list(pred_sets)



    def _pred_set_single(self, test_input, test_idx, best_model, marginal=True):
        '''
        Calculate the split conformal prediction sets for a single test point
        '''
        pred_set = []
        for label in range(self.n_classes):
            model_path = best_model[label]

            try:
                model_idx = self.model_list.index(model_path)
            except ValueError:
                print('Can not find the best model from the model list.')
                raise

            self.net.load_state_dict(th.load(model_path))

            p_hat_test = self.net.predict_prob(test_input.to(self.device))

            rng = np.random.default_rng(min(self.random_state*100000 + test_idx + label*10000, sys.maxsize))
            epsilon = rng.uniform(low=0.0, high=1.0, size=1)

            grey_box = ProbAccum(p_hat_test)

            alpha_new = self.alpha_calibrated[model_idx] if marginal else \
                        self.alpha_calibrated_lc[model_idx][label]
            S_hat = grey_box.predict_sets(alpha_new, epsilon=epsilon)


            if label in S_hat[0]:
                pred_set.append(label)

        return pred_set


class Conformal_PVals:
    '''
    Class for computing conformal p-values for any test set
    '''
    def __init__(self, net, device, cal_loader, model_list,
                 verbose = True, progress = True, random_state = 2023) -> None:
        self.net = net
        self.device = device
        self.cal_loader = cal_loader
        self.model_list = model_list
        self.verbose = verbose
        self.progress = progress
        self.random_state = random_state

        if self.verbose:
            print('Calibrating each model in the list...')
            sys.stdout.flush()
        self._calibrate_scores()
        if self.verbose:
            print('Initialization done!')
            sys.stdout.flush()




    def _calibrate_scores(self):
        n_model = len(self.model_list)
        self.cal_scores = [[]] * n_model

        if self.progress:
            iterator = tqdm(range(n_model))
        else:
            iterator = range(n_model)

        for model_idx in iterator:
            model_path = self.model_list[model_idx]
            self.net.load_state_dict(th.load(model_path))

            scores = []
            for inputs, _ in self.cal_loader:
                scores += self.net.get_anomaly_scores(inputs)

            self.cal_scores[model_idx] = scores



    def _compute_pval_single(self, test_input, best_model, left_tail):
        '''
        Calculate the conformal p-value for a single test point
        '''
        try:
            model_idx = self.model_list.index(best_model)
        except ValueError:
            print('Can not find the best model from the model list.')
            raise

        self.net.load_state_dict(th.load(best_model))
        test_score = self.net.get_anomaly_scores(test_input)
        cal_scores = self.cal_scores[model_idx]
        n_cal = len(cal_scores)

        if left_tail:
            pval = (1.0 + np.sum(np.array(cal_scores) <= np.array(test_score))) / (1.0 + n_cal)
        else:
            pval = (1.0 + np.sum(np.array(cal_scores) >= np.array(test_score))) / (1.0 + n_cal)
        return pval

    def compute_pvals(self, test_inputs, best_model, left_tail = False):
        """ Compute the conformal p-values for test points using a calibration set
        """
        n_test = len(test_inputs)
        n_model = len(best_model)
        assert n_test == n_model, 'Number of test points and number of models must match! If you want to use the same model for \
                                   all points, reshape the model input as a repeated list.'

        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        pvals = -np.zeros(n_test)
        for i in iterator:
            pvals[i] = self._compute_pval_single(test_inputs[i], best_model[i], left_tail)

        if self.verbose:
            print("Finished computing p-values for {} test points.".format(n_test))
            sys.stdout.flush()
        return list(pvals)


class Conformal_PI:
    '''
    Class for computing prediction intervals for regression.
    '''
    def __init__(self, device, cal_loader, alpha, net_quantile = None, net = None, 
                 verbose = True, progress = True, y_hat_min=None, y_hat_max=None) -> None:

        self.net = net 
        self.net_lower = net_quantile
        self.net_higher = net_quantile
        self.device = device
        self.cal_loader = cal_loader
        self.alpha = alpha
        self.verbose = verbose
        if y_hat_min is None:
            self.y_hat_min = -np.inf
        else:
            self.y_hat_min = y_hat_min
        if y_hat_max is None:
            self.y_hat_max = np.inf
        else:
            self.y_hat_max = y_hat_max

    def nonconformity_scores(self, output, target) -> float:
        """ Absolute residual as non-conformity score, used for mean regression
        """
        err = np.abs(output - target)
        return err 

    def score_CQR(self, output_lower, output_higher, target) -> float: 
        """ Nonconformity score for quantile regression
        """
        error_low = output_lower - target
        error_high = target - output_higher
        err = np.maximum(error_high, error_low)
        return err

    # def apply_inverse(self, nc):
    #     nc = np.sort(nc)
    #     index = int(np.ceil((1 - self.alpha) * (nc.shape[0] + 1))) - 1
    #     index = min(max(index, 0), nc.shape[0] - 1)
    #     return nc[index]

    def benchmark_CQR(self, test_inputs, bm_best_models_lower, bm_best_models_higher):

      assert self.net_lower is not None, "input a deep quantile regression net"
      assert self.net_higher is not None, "input a deep quantile regression net"
      
      self.net_lower.load_state_dict(th.load(bm_best_models_lower))
      self.net_higher.load_state_dict(th.load(bm_best_models_higher))

      cal_scores = []
      qc_count = 0

      for input, response in self.cal_loader:
      #input, response = input.to(self.device), response.to(self.device)
          with th.no_grad():
              # make prediction
              pred_lower = th.clip(self.net_lower(input)[0][0], self.y_hat_min, self.y_hat_max)
              pred_higher = th.clip(self.net_higher(input)[0][1], self.y_hat_min, self.y_hat_max)
              # quantile-crossing
              if pred_lower > pred_higher:
                qc_count += 1
                pred_temp = pred_lower 
                pred_lower = pred_higher
                pred_higher = pred_temp

              score = self.score_CQR(output_lower = pred_lower, output_higher = pred_higher, target = response)
              cal_scores.append(score.data.numpy())

      cal_scores = [j for i in cal_scores for j in i]
      n_cal = len(cal_scores)
      # Get the score quantile
      qhat = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-self.alpha))/n_cal, interpolation ='lower')
      

      test_pred_lower = th.clip(self.net_lower(test_inputs)[0][0], self.y_hat_min, self.y_hat_max)
      test_pred_higher = th.clip(self.net_higher(test_inputs)[0][1], self.y_hat_min, self.y_hat_max)
      # quantile-crossing
      if test_pred_lower > test_pred_higher:
        qc_count += 1
        pred_temp = test_pred_lower 
        test_pred_lower = test_pred_higher
        test_pred_higher = pred_temp
      print('quantile crossing occured {} times'.format(qc_count))
      # construct prediction interval for the test point
      pi = [Interval(test_pred_lower - qhat,test_pred_higher + qhat)]
      return pi

    
    def CES_CQR(self, test_inputs, best_models_lower, best_models_higher, method = ['union','cvxh'], no_empty = False, bm_best_models_higher = None, bm_best_models_lower = None):
        """Compute conformal prediction intervals for test inputs using the CES methods

        Args:
            test_inputs (tensor): test covariates
            best_model_lower (str): list of dictionaries containing information of the best model for lower quantile 
            best_model_higher (str): list of dictionaries containing information of the best model for higher quantile 
            bm_best_models_higher/bm_best_models_lower (str): benchmark best models (used when returning empty set)
            method (str): whether to take union or convex hull when forming the final prediction intervals
            no_empty (bool): whether to avoid empty prediction interval

        Returns:
            list of intervals : list of prediction interval
            format of the output (Example) : [Interval(2.5 , 3)]
        """

        
        assert method in ['union','cvxh'], "choose a method from 'union' or 'cvxh'"
        assert self.net_lower is not None, "input a deep quantile regression net"
        assert self.net_higher is not None, "input a deep quantile regression net"

        def concat_models(best_models_lower, best_models_higher): 
          '''Combine all unique knots from both lower quantile models and higher quantile models, so that within each knot interval, 
          there exist a unique best model for the lower quantile and a unique best model for the higher quantile
          '''
          best_models_lower_cp = deepcopy(best_models_lower)
          best_models_higher_cp = deepcopy(best_models_higher)

          for x in best_models_lower_cp:
            x["model_lower"] = x.pop("model")
          for x in best_models_higher_cp:
            x["model_higher"] = x.pop("model")
          
          concat_models = []
          
          curr_lower = 0
          curr_higher = 0

          n = len(set([i['knot_lower'] for i in best_models_lower_cp] + [i['knot_lower'] for i in best_models_higher_cp]))
            
          concat_models.append({'model_lower': best_models_lower_cp[0]['model_lower'], 
                                  'model_higher': best_models_higher_cp[0]['model_higher'], 
                                  'knot_lower': -np.inf,
                                  'knot_upper': min(best_models_lower_cp[0]['knot_upper'], best_models_higher_cp[0]['knot_upper'])})
          for i in range(n-1): 
            if best_models_lower_cp[curr_lower]['knot_upper'] <= best_models_higher_cp[curr_higher]['knot_upper']: 
              curr_lower += 1

              concat_models.append({'model_lower': best_models_lower_cp[curr_lower]['model_lower'], 
                              'model_higher': best_models_higher_cp[curr_higher ]['model_higher'], 
                              'knot_lower': best_models_lower_cp[curr_lower]['knot_lower'],
                              'knot_upper': min(best_models_lower_cp[curr_lower]['knot_upper'],best_models_higher_cp[curr_higher]['knot_upper'])})
            else: 
              curr_higher += 1
              concat_models.append({'model_lower': best_models_lower_cp[curr_lower]['model_lower'], 
                              'model_higher': best_models_higher_cp[curr_higher]['model_higher'], 
                              'knot_lower': best_models_higher_cp[curr_higher]['knot_lower'],
                              'knot_upper': min(best_models_lower_cp[curr_lower]['knot_upper'],best_models_higher_cp[curr_higher]['knot_upper'])})
              
          return concat_models 


        concat_results = concat_models(best_models_lower, best_models_higher)

        # store the final prediction intervals
        model_pi = []
        qc_count = 0

        for idx, result in enumerate(concat_results):
          self.net_lower.load_state_dict(th.load(result['model_lower']))
          self.net_higher.load_state_dict(th.load(result['model_higher']))

          cal_scores = []

          for input, response in self.cal_loader:
          #input, response = input.to(self.device), response.to(self.device)
              with th.no_grad():
                  # make prediction
                  pred_lower = th.clip(self.net_lower(input)[0][0], self.y_hat_min, self.y_hat_max)
                  pred_higher = th.clip(self.net_higher(input)[0][1], self.y_hat_min, self.y_hat_max)
                  if pred_lower > pred_higher:
                    qc_count += 1
                    pred_temp = pred_lower 
                    pred_lower = pred_higher
                    pred_higher = pred_temp

                  score = self.score_CQR(pred_lower, pred_higher, response)
                  cal_scores.append(score.data.numpy())

          cal_scores = [j for i in cal_scores for j in i]
          n_cal = len(cal_scores)
          # Get the score quantile
          qhat = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-self.alpha))/n_cal, interpolation ='lower')

          # with th.no_grad():
          test_pred_lower = th.clip(self.net_lower(test_inputs)[0][0], self.y_hat_min, self.y_hat_max)
          test_pred_higher = th.clip(self.net_higher(test_inputs)[0][1], self.y_hat_min, self.y_hat_max)
          if test_pred_lower > test_pred_higher:
            qc_count += 1
            pred_temp = test_pred_lower 
            test_pred_lower = test_pred_higher
            test_pred_higher = pred_temp
      
          # # construct prediction interval for the test point
          # take overlap between the original prediction intervals and the knots intervals
          truncated_pi = [max(test_pred_lower - qhat, result['knot_lower']), min(test_pred_higher + qhat, result['knot_upper'])]

          # drop the invalid prediction intervals
          if truncated_pi[0] > truncated_pi[1]:
              # print('drop non-overlapping set')
              continue

          model_pi.append(truncated_pi)

        print('quantile crossing occured {} times'.format(qc_count))

        if no_empty and len(model_pi) == 0:
          assert bm_best_models_higher is not None, "need to input a benchmark best model for deep quantile regression!"
          assert bm_best_models_lower is not None, "need to input a benchmark best model for deep quantile regression!"

          pi = self.benchmark_CQR(test_inputs, bm_best_models_lower, bm_best_models_higher)
          print('avoided empty interval, final pi is {}'.format(pi))
          return pi

        if method == 'union':
          def union(data):
              """ Union of a list of intervals """
              intervals = [Interval(l, u) for (l, u) in data]
              uni = Union(*intervals)
              return [list(uni.args[:2])] if isinstance(uni, Interval) else list(uni.args)

          # take union of the list of intervals
          unioned_pi = union(model_pi)

          if len(unioned_pi)==1:
              # convert the format into Interval
              unioned_pi = [Interval(unioned_pi[0][0],unioned_pi[0][1])]
          return unioned_pi


        elif method == 'cvxh':
          l = min(model_pi, key=lambda x: x[0])[0]
          u = max(model_pi, key=lambda x: x[1])[1]
          return [Interval(l,u)]



    def benchmark_ICP(self, test_inputs, bm_best_model):
        """Compute conformal prediction intervals for test inputs using the benchmark methods

        Args:
            test_inputs (tensor): test covariates
            best_model (str): path of the best model

        Returns:
            list of intervals : list of prediction interval
            format of the output (Example) : [Interval(2.5 , 3)]
        """
        assert self.net is not None, "need to input a deep neural net trained minimizing MSE"
        self.net.load_state_dict(th.load(bm_best_model))

        cal_scores = []
        for X_batch, y_batch in self.cal_loader:
            # input, response = input.to(self.device), response.to(self.device)
            with th.no_grad():
                # make prediction using the model trained
                pred = th.clip(self.net(X_batch), self.y_hat_min, self.y_hat_max)
                # compute the conformity scores using predictions and the real calibration response
                score = self.nonconformity_scores(pred, y_batch)
                cal_scores.append(score.data.numpy())


        n_cal = len(cal_scores)
        # Get the score quantile
        qhat = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-self.alpha))/n_cal, interpolation='higher')

        # with th.no_grad():
        test_pred = th.clip(self.net(test_inputs), self.y_hat_min, self.y_hat_max)

        # construct prediction interval for the test point
        pi = [Interval(test_pred - qhat,test_pred + qhat)]

        return pi



    def CES_icp(self, test_inputs, best_models, method = ['union','cvxh'], no_empty = False, mod = None):
        """Compute conformal prediction intervals for test inputs using the CES methods

        Args:
            test_inputs (tensor): test covariates
            best_model (str): list of dictionaries containing information of the best model
            method (str): whether to take union or convex hull when forming the final prediction intervals
            no_empty (bool): whether to avoid empty prediction interval

        Returns:
            list of intervals : list of prediction interval
            format of the output (Example) : [Interval(2.5 , 3)]
        """

        assert method in ['union','cvxh'], "choose a method from 'union' or 'cvxh'"
        assert self.net is not None, "need to input a deep neural net trained minimizing MSE"

        # store the final prediction intervals
        model_pi = []

        for bm in best_models:
          self.net.load_state_dict(th.load(bm['model']))

          # store the empirical conformity scores
          cal_scores = []

          for input, response in self.cal_loader:
          #input, response = input.to(self.device), response.to(self.device)

              with th.no_grad():
                  # make prediction
                  pred = th.clip(self.net(input), self.y_hat_min, self.y_hat_max)
              # compute the conformity scores
              score = self.nonconformity_scores(pred, response)
              cal_scores.append(score.data.numpy())

          # unpack calibration scores into a list of numbers
          cal_scores = [j for i in cal_scores for j in i]
          n_cal = len(cal_scores)
          # Get the score quantile
          qhat = np.quantile(cal_scores, np.ceil((n_cal+1)*(1-self.alpha))/n_cal, interpolation='higher')
          test_pred = th.clip(self.net(test_inputs), self.y_hat_min, self.y_hat_max).data.numpy()
          # original prediction intervals constructed in a standard ICP way
          initial_pi = [test_pred - qhat, test_pred + qhat]

          # take overlap between the original prediction intervals and the knots intervals
          truncated_pi = [max(initial_pi[0], bm['knot_lower']), min(initial_pi[1], bm['knot_upper'])]

          # drop the invalid prediction intervals
          if truncated_pi[0] > truncated_pi[1]:
              continue

          model_pi.append(truncated_pi)

        if no_empty and len(model_pi) == 0:
          assert mod is not None, "need to input a benchmark best model!"
          _, bm_best_model, _ = mod.select_model(test_inputs = None)
          pi = self.benchmark_ICP(test_inputs, bm_best_model)
          print('avoided empty interval, final pi is {}'.format(pi))
          return pi

        if len(model_pi) == 0: 
            return [EmptySet]
        
        if method == 'union':
          def union(data):
              """ Union of a list of intervals """
              intervals = [Interval(l, u) for (l, u) in data]
              uni = Union(*intervals)
              return [list(uni.args[:2])] if isinstance(uni, Interval) else list(uni.args)

          # take union of the list of intervals
          unioned_pi = union(model_pi)

          if len(unioned_pi)==1:
              # convert the format into Interval
              unioned_pi = [Interval(unioned_pi[0][0],unioned_pi[0][1])]
          return unioned_pi


        elif method == 'cvxh':
          l = min(model_pi, key=lambda x: x[0])[0]
          u = max(model_pi, key=lambda x: x[1])[1]
          return [Interval(l,u)]
      
