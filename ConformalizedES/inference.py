import time
import os
import torch as th
import numpy as np
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
            pred_sets.append(self._pred_set_single(test_inputs[i], i, best_model[i], marginal))
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
