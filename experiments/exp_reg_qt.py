from __future__ import print_function, division

import itertools
import time

import torch as th
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler
from sympy import *
import pathlib
import pdb
import matplotlib.pyplot as plt
import sys
import tempfile
from datasets import datasets
import models

from scipy.stats.mstats import mquantiles

from sklearn.datasets import make_friedman1, make_regression

import __main__ as main

sys.path.append('..')
from third_party.classification import *
from ConformalizedES.method import CES_regression
from ConformalizedES.networks import quantreg_model, AllQuantileLoss
from ConformalizedES.inference import Conformal_PI
from ConformalizedES import theory
from third_party.coverage import *

##############
# Parameters #
##############

conf = 1

# Default parameters
#data = "regression"
data = "friedman1"
method = "naive"
n = 1000
n_features = 200
noise = 100
seed = 2022

# Input
if True:
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 10:
        print("Error: incorrect number of parameters.")
        quit()
    sys.stdout.flush()

    data = sys.argv[1]
    method = sys.argv[2]
    n = int(sys.argv[3])
    n_features = int(sys.argv[4])
    noise = float(sys.argv[5]) / 100
    lr = float(sys.argv[6])
    wd = float(sys.argv[7])
    batch_size = int(sys.argv[9])
    # dropout = float(sys.argv[10])
    seed = int(sys.argv[8])
    # clip = sys.argv[11]
    


# Fixed data parameters
n_test = 1000

# Training hyperparameters
dropout = 0
# num_epochs = 1000
clip = "no_clip"
# batch_size = 25
# dropout = 0.1
num_epochs = 2000
hidden_layer_size = 128
optimizer_alg = 'adam'



if (method=="ces"):
    save_every = 10     # Save model after every few epoches
else:
    save_every = 10     # Save model after every few epoches


# Other parameters
show_plots = True
num_cond_coverage = 1

# Parse input arguments

# Output file
outfile_prefix = "exp"+str(conf) + "/" + "exp" + str(conf) + "_" + str(data) + "_" + method + "_n" + str(n)
outfile_prefix += "_p" + str(n_features) + "_noise" + str(int(noise*100)) + "_lr" + str(lr) + "_wd" + str(wd) + "_batchsize" + str(batch_size) + "_seed" + str(seed)
print("Output file: {:s}.".format("results/"+outfile_prefix), end="\n")


####################
# Useful functions #
####################

class PrepareData(Dataset):

    def __init__(self, X, Y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(Y):
            self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].float(), self.Y[idx].float()


def plot_loss(train_loss, val_loss, test_loss=None, out_file=None):
    x = np.arange(1, len(train_loss) + 1)

    # Colors from Colorbrewer Paired_12
    colors = [[31, 120, 180], [51, 160, 44], [250,159,181]]
    colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]

    plt.figure()
    plt.plot(x, train_loss, color=colors[0], label="Training loss", linewidth=2)
    plt.plot(x, val_loss, color=colors[1], label="Validation loss", linewidth=2)
    if test_loss is not None:
        plt.hlines(test_loss, np.min(x), np.max(x), color="red")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the training, validation and test loss")

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')

    plt.show()

def make_dataset(n_samples=1, n_features=10, noise=0, random_state=2022):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(size=(n_samples,n_features))
    epsilon = rng.normal(loc=0.0, scale=1.0, size=(n_samples,))
    beta = np.zeros((n_features,1))
    beta[0:5] = 1
    def f(x):
        return 2 * np.sin(np.pi*x) + np.pi*x
    Z = np.dot(X,beta).flatten()
    Y = f(Z) + epsilon
    return X, Y

def load_dataset(dataname, n_samples=1, n_features=10, noise=0, random_state=2022):
    base_dataset_path = "datasets/"
    X_all, Y_all = datasets.GetDataset(dataname, base_dataset_path)
    rng = np.random.default_rng(random_state)
    if n_samples > len(Y_all):
        n_samples = len(Y_all)
    idx = rng.choice(len(Y_all), size=(n_samples,))
    X = X_all[idx]
    Y = Y_all[idx]
    return X, Y




#####################
# Generate the data #
#####################

# Set random seeds
np.random.seed(seed)
th.manual_seed(seed)

n_cal = np.round(n*0.25).astype(int)
n_es = n_cal
n_samples_tot = n + n_test

if data=="friedman1":
    X_all, Y_all = make_friedman1(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)

elif data=="regression":
    X_all, Y_all = make_regression(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)

elif data=="chr":
    data_model = models.Model_Ex3(p=n_features)
    X_all = data_model.sample_X(n_samples_tot)
    Y_all = data_model.sample_Y(X_all)

elif data=="custom":
    X_all, Y_all = make_dataset(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)
    _, X_all, _, Y_all = train_test_split(X_all, Y_all, test_size=n_samples_tot, random_state=seed)

else:
    X_all, Y_all = load_dataset(data, n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)
    Y_all = Y_all.flatten()
    if n_samples_tot < len(Y_all):
        _, X_all, _, Y_all = train_test_split(X_all, Y_all, test_size=n_samples_tot, random_state=seed)

if n_samples_tot > len(Y_all):
    n = len(Y_all) - n_test
    n_cal = np.round(n*0.25).astype(int)
    n_es = n_cal
    n_samples_tot = n + n_test

# Scale the data
# if False:
#     X_all = StandardScaler().fit_transform(X_all)
#     Y_all = StandardScaler().fit_transform(Y_all.reshape((len(Y_all),1))).flatten()

# Find approximate marginal quantiles of Y
y_hat_min, y_hat_max = mquantiles(Y_all, [0.05,0.95])


# Set test data aside
X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=n_test, random_state=seed)


# scale the labels by dividing each by the mean absolute response
mean_y_train = np.mean(np.abs(Y))
Y = np.squeeze(Y)/mean_y_train
Y_test = np.squeeze(Y_test)/mean_y_train

# zero mean and unit variance scaling 
X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)
print('Standardizing data...')



##################
# Split the data #
##################

if method=="benchmark":
    # Separate training data
    X_train, X_escal, Y_train, Y_escal = train_test_split(X, Y, test_size=(n_es + n_cal), random_state=seed)
    # Separate es data
    X_es, X_cal, Y_es, Y_cal = train_test_split(X_escal, Y_escal, test_size=n_cal, random_state=seed)
    n_train = len(Y_train)

elif (method=="theory") or (method=="naive") or (method=="ces"):
    # Separate training data
    X_train, X_escal, Y_train, Y_escal = train_test_split(X, Y, test_size=(n_cal), random_state=seed)
    X_es = X_escal
    Y_es = Y_escal
    X_cal = X_escal
    Y_cal = Y_escal
    n_train = len(Y_train)

else:
    print("Unknown method!")
    sys.stdout.flush()
    exit(-1)



print("Data size: train (%d, %d), early-stopping (%d, %d), calibration (%d, %d), test (%d, %d)." % \
      (X_train.shape[0], X_train.shape[1],
       X_es.shape[0], X_es.shape[1],
       X_cal.shape[0], X_cal.shape[1],
       X_test.shape[0], X_test.shape[1]))
sys.stdout.flush()

###################
# Train the model #
###################

train_loader = DataLoader(PrepareData(X_train, Y_train), batch_size=batch_size)
es_loader = DataLoader(PrepareData(X_es, Y_es), batch_size=batch_size, drop_last = True)
calib_loader = DataLoader(PrepareData(X_cal, Y_cal), batch_size=1, shuffle = False, drop_last=True)
test_loader = DataLoader(PrepareData(X_test, Y_test), batch_size= 1, shuffle = False)

# intialize the model
in_shape = X_train.shape[1]
# intialize the model
quantiles_net = [0.05, 0.95]

# intialize the model

# intialize the model
mod = quantreg_model(quantiles = quantiles_net, in_shape = in_shape, hidden_size = hidden_layer_size, dropout = dropout )
pinball_loss = AllQuantileLoss(quantiles_net)


if optimizer_alg == 'adam':
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr, weight_decay = wd)
else:
    optimizer = torch.optim.SGD(mod.parameters(), lr=lr, weight_decay = wd)

if th.cuda.is_available():
    # Make CuDNN Determinist
    th.backends.cudnn.deterministic = True
    th.cuda.manual_seed(seed)

# Define default device, we should use the GPU (cuda) if available
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# initialization
qt_reg = CES_regression(mod, device, 
                         train_loader, 
                         batch_size=batch_size, 
                         max_epoch = num_epochs, 
                         learning_rate=lr, 
                         val_loader=es_loader, 
                         criterion= pinball_loss, 
                         optimizer=optimizer,
                         verbose = False)


# Train the model and save snapshots
tmp_dir = tempfile.TemporaryDirectory().name
print("Saving models in {}".format(tmp_dir))
sys.stdout.flush()
qt_reg.full_train(save_dir = tmp_dir, save_every = save_every)

#############################
# Apply conformal inference #
#############################

def apply_conformal(selected_model_lower, selected_model_higher):
    results = pd.DataFrame({})

    for alpha in [0.1]:

        # store coverage indicator for every test sample
        coverage_BM = []
        # store size of the prediction interval
        size_BM = []
        # store test loss
        test_losses_BM = []
        # store prediction intervals for every test sample
        pi_BM = []

        # initialize
        if method=="theory":
            T = len(qt_reg.model_list)
            alpha_2 = theory.inv_hybrid(T, n_cal, alpha)
            alpha_2 = np.clip(alpha_2, 1.0/n_cal, 1)
        else:
            alpha_2 = alpha
        
        if clip == 'no_clip':
            C_PI = Conformal_PI(device, calib_loader, alpha_2, net_quantile = mod)#, y_hat_min=y_hat_min, y_hat_max=y_hat_max)
        else: 
            C_PI = Conformal_PI(device, calib_loader, alpha_2, net_quantile = mod, y_hat_min=y_hat_min, y_hat_max=y_hat_max)
            

        print("Applying conformal prediction...")
        sys.stdout.flush()

        for input, response in tqdm(test_loader):
            # find prediction interval
            if (method=="benchmark") or (method=="naive") or (method=="theory"):
                ci_method = C_PI.benchmark_CQR(input, selected_model_lower, selected_model_higher)
                pi_BM.append(ci_method)
                # find size and coverage indicator
                size_BM.append(ci_method[0]._measure)
                coverage_BM.append(response in ci_method[0])                
                
            else:
                best_models_lower, best_models_higher = qt_reg.select_model(input, method = "quantile")
                ci_method = C_PI.CES_CQR(input, best_models_lower, best_models_higher, method = 'cvxh')
                pi_BM.append(ci_method)
                coverage_single = sum([response in intv for intv in ci_method]) > 0
                coverage_BM.append(coverage_single)
                size_single = sum([intv._measure for intv in ci_method])
                size_BM.append(size_single)
                


            # evaluate the out of sample losses
            ## load the best model
            reg_lower_tmp = CES_regression(mod, device, train_loader, batch_size=batch_size, max_epoch = num_epochs, learning_rate=lr, 
                                       val_loader=es_loader, criterion= pinball_loss, optimizer=optimizer,verbose = True)
            reg_higher_tmp = CES_regression(mod, device, train_loader, batch_size=batch_size, max_epoch = num_epochs, learning_rate=lr, 
                                       val_loader=es_loader, criterion= pinball_loss, optimizer=optimizer,verbose = True)
            reg_lower_tmp.net.load_state_dict(torch.load(selected_model_lower, map_location=device))
            reg_higher_tmp.net.load_state_dict(torch.load(selected_model_higher, map_location=device))
            ## compute loss on test samples
            test_loss = reg_lower_tmp.get_loss(input, response)
            test_loss = test_loss[0]
            test_losses_BM.append(test_loss)


        print("Evaluating conditional coverage...")
        sys.stdout.flush()
        # store conditional coverage
        wsc_coverages_BM = []

        # compute conditional coverage
        for i in tqdm(np.arange(num_cond_coverage)):
            wsc_coverage = wsc_unbiased(X_test, Y_test, pi_BM, M=100, delta = 0.1, test_size = 0.5)
            wsc_coverages_BM.append(wsc_coverage)

        ################
        # Save results #
        ################

        marg_coverage = np.mean(coverage_BM)
        cond_coverage = np.mean(wsc_coverages_BM)
        avg_size = np.mean(size_BM)
        test_loss = np.mean(test_losses_BM)

        res = pd.DataFrame({
            'data' : [data],
            'method' : [method],
            'n' : [n],
            'n_train' : [n_train],
            'n_cal' : [n_cal],
            'n_features' : [n_features],
            'n_test' : [n_test],
            'noise' : [noise],
            'lr' : [lr],
            'wd' : [wd],
            'batchs_zie':[batch_size],
            'seed' : [seed],
            'alpha' : [alpha],
            'alpha_2' : [alpha_2],
            'marg_coverage' : [marg_coverage],
            'cond_coverage' : [cond_coverage],
            'avg_size' : [avg_size],
            'test_loss' : [test_loss],
            'best_epoch_lower' : [best_epoch_lower],
            'best_epoch_higher' : [best_epoch_higher],
            'optimizer' : [optimizer_alg]
        })

        results = pd.concat([results, res])

        if (show_plots) and (alpha==0.1):
            plot_loss(qt_reg.train_loss_history[5:], qt_reg.val_loss_history[5:], test_loss = test_loss, out_file="plots/"+outfile_prefix+".png")


    return results

#########################
# Conformal inference   #
#########################

# compute validation loss (burn-out first 20 epochs)
val_loss_higher = qt_reg.sep_val_loss_history[quantiles_net[1]][-20:]
val_loss_lower = qt_reg.sep_val_loss_history[quantiles_net[0]][-20:]

# Test the best model
bm_loss_lower, bm_model_lower, val_loss_history_lower, bm_loss_higher, bm_model_higher, val_loss_history_higher = qt_reg.select_model(method = 'quantile')
best_epoch_lower = np.argmin(val_loss_history_lower) + 1
best_epoch_higher = np.argmin(val_loss_history_higher) + 1

results_best = apply_conformal(bm_model_lower, bm_model_higher)

if (method != "ces"):
    # Test the last model
    full_model = qt_reg.model_list[-1]
    results_full = apply_conformal(full_model, full_model)
    results_full["method"] = results_full["method"] + "-full"

    # Combine results
    results = pd.concat([results_best, results_full])
else:
    results = results_best

print("\nResults:")
sys.stdout.flush()
print(results)
sys.stdout.flush()


################
# Save results #
################
outfile = "results/" + outfile_prefix + ".txt"
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()
