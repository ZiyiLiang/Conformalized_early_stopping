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

from sklearn.datasets import make_friedman1, make_regression

import __main__ as main

sys.path.append('..')
from third_party.classification import *
from ConformalizedES.method import CES_regression
from ConformalizedES.networks import mse_model, MSE_loss
from ConformalizedES.inference import Conformal_PI
from third_party.coverage import *

##############
# Parameters #
##############

conf = 1

# Default parameters
#data = "regression"
data = "friedman1"
method = "naive"
n_train = 1000
n_cal = 100
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
    n_train = int(sys.argv[3])
    n_cal = int(sys.argv[4])
    n_features = int(sys.argv[5])
    noise = float(sys.argv[6]) / 100
    lr = float(sys.argv[7])
    wd = float(sys.argv[8])
    seed = int(sys.argv[9])


# Fixed data parameters
n_test = 100

# Training hyperparameters
batch_size = 50
dropout = 0
num_epochs = 1000
hidden_layer_size = 100
optimizer_alg = 'adam'

if (method=="ces"):
    save_every = 10    # Save model after every few epoches
else:
    save_every = 1     # Save model after every few epoches
    

# Other parameters
show_plots = True
num_cond_coverage = 1

# Parse input arguments

# Output file
outfile_prefix = "exp"+str(conf) + "/" + "exp" + str(conf) + "_" + str(data) + "_" + method + "_n" + str(n_train) + "_n" + str(n_cal)
outfile_prefix += "_p" + str(n_features) + "_noise" + str(int(noise*100)) + "_lr" + str(lr) + "_wd" + str(wd) + "_seed" + str(seed)
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

    #plt.show()

def make_dataset(n_samples=1, n_features=10, noise=0, random_state=2022):
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples,n_features))
    epsilon = rng.normal(loc=0.0, scale=1.0, size=(n_samples,))
    Y = ((X[:,0]<-1.5)+(X[:,0]>1.5)) + ((X[:,1]<-1.5)+(X[:,1]>1.5)) * ((X[:,2]>-2)*(X[:,2]<2)) + X[:,3] ** 2 - 0.1 * X[:,4] ** 3
    Y = ( Y  + noise * epsilon ).astype(float) + noise * epsilon
    return X, Y



#####################
# Generate the data #
#####################

# Set random seeds
np.random.seed(seed)
th.manual_seed(seed)

n_es = n_cal
n_samples_tot = n_train + n_es + n_cal + n_test

if data=="friedman1":
    X_all, Y_all = make_friedman1(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)
elif data=="regression":
    X_all, Y_all = make_regression(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)
elif data=="custom":
    X_all, Y_all = make_dataset(n_samples=n_samples_tot, n_features=n_features, noise=noise, random_state=seed)
else:
    print("Unknown data distribution!")
    sys.stdout.flush()

# Scale the data
X_all = StandardScaler().fit_transform(X_all)
Y_all = StandardScaler().fit_transform(Y_all.reshape((len(Y_all),1))).flatten()

# Set test data aside
X, X_test, Y, Y_test = train_test_split(X_all, Y_all, test_size=n_test, random_state=seed)

##################
# Split the data #
##################

if method=="benchmark":
    # Separate training data
    X_train, X_escal, Y_train, Y_escal = train_test_split(X, Y, test_size=(n_es + n_cal), random_state=seed)
    # Separate es data
    X_es, X_cal, Y_es, Y_cal = train_test_split(X_escal, Y_escal, test_size=n_cal, random_state=seed)

elif (method=="naive") or (method=="ces"):
    # Separate training data
    X_train, X_escal, Y_train, Y_escal = train_test_split(X, Y, test_size=(n_cal), random_state=seed)
    X_es = X_escal
    Y_es = Y_escal
    X_cal = X_escal
    Y_cal = Y_escal

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
mod = mse_model(in_shape = in_shape, hidden_size = hidden_layer_size)
if optimizer_alg == 'adam':
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr, betas=(0,0.1), weight_decay = wd)
else:
    optimizer = torch.optim.SGD(mod.parameters(), lr=lr, weight_decay = wd)

if th.cuda.is_available():
    # Make CuDNN Determinist
    th.backends.cudnn.deterministic = True
    th.cuda.manual_seed(seed)

# Define default device, we should use the GPU (cuda) if available
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# initialization
reg_model = CES_regression(mod, device, train_loader, batch_size=batch_size, max_epoch = num_epochs,
                           learning_rate=lr, val_loader=es_loader, criterion= MSE_loss,
                           optimizer=optimizer, verbose = True)

# Train the model and save snapshots
tmp_dir = tempfile.TemporaryDirectory().name
print("Saving models in {}".format(tmp_dir))
sys.stdout.flush()
reg_model.full_train(save_dir = tmp_dir, save_every = save_every)


#############################
# Apply conformal inference #
#############################

def apply_conformal(selected_model):
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
        C_PI = Conformal_PI(mod, device, calib_loader, alpha)

        print("Applying conformal prediction...")
        sys.stdout.flush()

        for input, response in tqdm(test_loader):
            # find prediction interval
            if (method=="benchmark") or (method=="naive"):
                ci_method = C_PI.benchmark_ICP(input, selected_model)
            else:
                best_models = reg_model.select_model(input)
                ci_method = C_PI.CES_icp(input, best_models, method = 'cvxh')

            pi_BM.append(ci_method)
            # find size and coverage indicator
            size_BM.append(ci_method[0]._measure)
            coverage_BM.append(response in ci_method[0])
            # evaluate the out of sample losses
            ## load the best model
            reg_model_tmp = CES_regression(mod, device, train_loader, batch_size=batch_size, max_epoch = num_epochs, learning_rate=lr, 
                                           val_loader=es_loader,
                                           verbose = False, criterion = MSE_loss, optimizer = optimizer)
            reg_model_tmp.net.load_state_dict(torch.load(selected_model, map_location=device))
            ## compute loss on test samples
            test_loss = reg_model_tmp.get_loss(input, response)
            test_losses_BM.append(test_loss)


        print("Evaluating conditional coverage...")
        sys.stdout.flush()
        # store conditional coverage
        wsc_coverages_BM = []

        # compute conditional coverage
        for i in tqdm(np.arange(num_cond_coverage)):
            wsc_coverage = wsc_unbiased(X_test, Y_test, pi_BM, M=100, delta = 0.1)
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
            'n_train' : [n_train],
            'n_cal' : [n_cal],
            'n_features' : [n_features],
            'n_test' : [n_test],
            'noise' : [noise],
            'lr' : [lr],
            'wd' : [wd],
            'seed' : [seed],
            'alpha' : [alpha],
            'marg_coverage' : [marg_coverage],
            'cond_coverage' : [cond_coverage],
            'avg_size' : [avg_size],
            'test_loss' : [test_loss],
            'best_epoch' : [best_epoch],
            'optimizer' : [optimizer_alg]
        })

        results = pd.concat([results, res])

        if (show_plots) and (alpha==0.2):
            plot_loss(reg_model.train_loss_history[5:], reg_model.val_loss_history[5:], test_loss = test_loss, out_file="plots/"+outfile_prefix+".png")

    return results

#########################
# Conformal inference   #
#########################

# compute validation loss (burn-out first 20 epochs)
val_loss = np.mean(reg_model.val_loss_history[-20:])

# Test the best model
bm_loss, bm_model, loss_history = reg_model.select_model()
best_epoch = np.argmin(loss_history) + 1
results_best = apply_conformal(bm_model)

if (method != "ces"):
    # Test the last model
    full_model = reg_model.model_list[-1]
    results_full = apply_conformal(full_model)
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
