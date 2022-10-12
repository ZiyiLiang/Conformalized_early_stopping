import time
import os
import torch as th
import numpy as np
import pdb
import pathlib
from tqdm import tqdm
from classification import ProbabilityAccumulator as ProbAccum
from scipy.stats.mstats import mquantiles



class Conformal_PSet:
    '''
    Class for computing marginal or label conditional conformal prediction sets.
    '''
    def __init__(self, net, device, cal_loader, n_classes, model_list, alpha, 
                 verbose = True, progress = True, random_state = 2023) -> None:
        self.net = net
        self.device = device
        self.cal_loader = cal_loader
        self.n_classes = n_classes
        self.model_list = model_list
        self.alpha = alpha
        self.verbose = verbose
        self.progress = progress
        self.random_state = random_state

        if self.verbose:
            print('Calibrating each model in the list...')
        self._calibrate_alpha()
        if self.verbose:
            print('Initialization done!')

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
        
        if self.progress:
            iterator = tqdm(range(n_test))
        else:
            iterator = range(n_test)

        pred_sets = []
        for i in iterator:
            pred_sets.append(self._pred_set_single(test_inputs[i][None], i, best_model[i], marginal))
        if True:
            print("Finished computing {} prediction sets for {} test points.".format(['label conditional', 'marginal'][marginal], n_test))
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

            rng = np.random.default_rng(self.random_state + test_idx + label*10000)
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
        self._calibrate_scores()
        if self.verbose:
            print('Initialization done!')
    

    
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
    


    # def _get_vt_scores_single(self, test_input, model_path):
    #     """ Get the scores of validation set and a single test point by a specific model
    #     """
    #     cal_scores = []

    #     self.net.load_state_dict(th.load(model_path))

    #     # Compute the anomaly scores for calibration set
    #     for inputs, _ in self.cal_loader:
    #         cal_scores.append(self.net.get_anomaly_scores(inputs))
    #     # Compute the anomaly score for the test point
    #     test_score = self.net.get_anomaly_scores(test_input)

    #     return cal_scores, test_score



    def _compute_pval_single(self, test_input, best_model):
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

        pval = (1.0 + np.sum(np.array(cal_scores) > np.array(test_score))) / (1.0 + n_cal)
        return pval

    def compute_pvals(self, test_inputs, best_model):
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
            pvals[i] = self._compute_pval_single(test_inputs[i], best_model[i])

        if self.verbose:
            print("Finished computing p-values for {} test points.".format(n_test))
        return list(pvals)

