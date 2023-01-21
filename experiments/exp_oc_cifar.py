import numpy as np
import sys, os
import pandas as pd
import torch as th
import torch.optim as optim
import shutil
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("../")
sys.path.append('../third_party')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ConformalizedES.method import CES_multiClass
from ConformalizedES.networks import SimpleConvolutionalNetwork
from ConformalizedES.inference import Conformal_PVals
from ConformalizedES.utils import eval_pvalues
from ConformalizedES import theory




#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 5:
        print("Error: incorrect number of parameters.")
        quit()

    n_data = int(sys.argv[1])
    lr = float(sys.argv[2])
    n_epoch = int(sys.argv[3])
    seed = int(sys.argv[4])



else: # Default parameters
    n_data = 500
    lr = 0.001
    n_epoch = 20
    seed = 0

# Fixed experiment parameters
batch_size = 64           # batch size for training set
n_test_samples = 100
save_every = 1         # Save every training snapshot  
alpha_list = [0.1]



###############
# Output file #
###############
outdir = "results/outlierDetect/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_data) + "_lr" + str(lr) + "_epoch" + str(n_epoch) +\
                 "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/outlierDetect/"+outfile_name+"/"

# Header for results file
def add_header(df):
    df["n_epoch"] = n_epoch
    df["n_data"] = n_data
    df["seed"] = seed
    df["lr"] = lr
    df["batch_size"] = batch_size
    return df



#################
# Download Data #
#################

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = len(classes)



class PrepareData(Dataset):

    def __init__(self, X, Y, transform):
        if len(X.shape) ==3:
            X = X[None]
        self.X = X
        self.Y = th.from_numpy(Y.astype('int64'))
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.Y[idx]



###############
# Split Data  #
###############

np.random.seed(seed)
th.manual_seed(seed)

# Benchmark data splitting: equally split the data into 3 sets
n_full = len(train_set)
n_es_bm = int(n_data*0.25)         # Make sure the calibration data and validation data only contains inliers
n_calib_bm = n_es_bm
n_train_bm = n_data-n_es_bm-n_calib_bm

X_all=train_set.data
Y_all=np.array(train_set.targets)

X_train_bm, X_bm, Y_train_bm, Y_bm = train_test_split(X_all, Y_all, train_size = n_train_bm, random_state= 10*seed)

idx_in= np.where(Y_bm == 3)[0]
X_in, Y_in = X_bm[idx_in], Y_bm[idx_in]

rng = np.random.default_rng(seed)
idx = rng.choice(len(Y_in), size=(n_es_bm + n_calib_bm,), replace = False)

X_es_bm, X_calib_bm, Y_es_bm, Y_calib_bm = train_test_split(X_in[idx], Y_in[idx], train_size = n_es_bm)


# CES data splitting: calibration set is not needed, merge back to the training set
n_escal_ces = n_calib_bm
n_train_ces = n_data-n_escal_ces

X_train_ces, X_ces, Y_train_ces, Y_ces = train_test_split(X_all, Y_all, train_size = n_train_ces, random_state= 10*seed+1)

idx_in= np.where(Y_ces == 3)[0]
X_in, Y_in = X_ces[idx_in], Y_ces[idx_in]
idx = rng.choice(len(Y_in), size=(n_escal_ces,), replace = False)

X_escal_ces, Y_escal_ces = X_in[idx], Y_in[idx]


# sample test data with half inliers and half outliers
n_test = 100
n_in = int(n_test*0.5)
n_out = n_test - n_in

X_test_all=test_set.data
Y_test_all=np.array(test_set.targets)

idx_testin, idx_testout = np.where(Y_test_all==3)[0], np.where(Y_test_all!=3)[0]
X_test_inliers, Y_test_inliers = X_test_all[idx_testin], Y_test_all[idx_testin]
X_test_outliers, Y_test_outliers = X_test_all[idx_testout], Y_test_all[idx_testout]

idx_in = rng.choice(len(Y_test_inliers), size=(n_in,), replace = False)
idx_out = rng.choice(len(Y_test_outliers), size=(n_out,), replace = False)

X_test = np.array(list(X_test_inliers[idx_in]) + list(X_test_outliers[idx_out]))
Y_test = np.array(list(Y_test_inliers[idx_in]) + list(Y_test_outliers[idx_out]))


print("Use {:d} training samples, {:d} early stopping samples, {:d} calibration samples for the Data splitting method."\
     .format(n_train_bm, n_es_bm, n_calib_bm))
print("Use {:d} training samples, {:d} early stopping samples, {:d} calibration samples for the CES method."\
     .format(n_train_ces, n_escal_ces, n_escal_ces))
sys.stdout.flush()



######################
# Create Dataloader  #
######################

train_loader_bm = DataLoader(PrepareData(X_train_bm, Y_train_bm, transform), batch_size=batch_size)
es_loader_bm = DataLoader(PrepareData(X_es_bm, Y_es_bm, transform), batch_size=int(n_es_bm/5))
calib_loader_bm = DataLoader(PrepareData(X_calib_bm, Y_calib_bm, transform), batch_size=int(n_calib_bm/5))

train_loader_ces = DataLoader(PrepareData(X_train_ces, Y_train_ces,transform), batch_size=batch_size)
escal_loader_ces = DataLoader(PrepareData(X_escal_ces, Y_escal_ces,transform), batch_size=int(n_escal_ces/5))

test_loader = DataLoader(PrepareData(X_test, Y_test, transform), batch_size= n_test, shuffle = False)

# get all test images
dataiter = iter(test_loader)
inputs, labels = dataiter.next()
is_outlier = labels!=3


################
# Train models #
################

if th.cuda.is_available():
    # Make CuDNN Determinist
    th.backends.cudnn.deterministic = True
    th.cuda.manual_seed(seed)

# Define default device, we should use the GPU (cuda) if available
device = th.device("cuda" if th.cuda.is_available() else "cpu")


#------------ Training with benchmark data splitting ------------------#

# create wrapper function to modify the criterion.
net_bm = SimpleConvolutionalNetwork()
Loss = th.nn.CrossEntropyLoss()
def criterion(outputs, inputs, targets):
    return Loss(outputs, targets)
optimizer_bm = optim.Adam(net_bm.parameters(), lr=lr)

CES_mc_bm = CES_multiClass(net_bm, device, train_loader_bm, n_classes=n_classes, batch_size=batch_size, max_epoch=n_epoch, 
                        learning_rate=lr, val_loader=es_loader_bm, criterion=criterion,optimizer=optimizer_bm)

CES_mc_bm.full_train(save_dir = modeldir+'benchmarks', save_every = save_every)


#------------ Training with without data splitting ------------------#

net_ces = SimpleConvolutionalNetwork()
Loss = th.nn.CrossEntropyLoss()
def criterion(outputs, inputs, targets):
    return Loss(outputs, targets)
optimizer_ces = optim.Adam(net_ces.parameters(), lr=lr)

# Initialize the CES class with model parameters
CES_mc_ces = CES_multiClass(net_ces, device, train_loader_ces, n_classes=n_classes, batch_size=batch_size, max_epoch=n_epoch, 
                        learning_rate=lr, val_loader=escal_loader_ces, criterion=criterion,optimizer=optimizer_ces)

CES_mc_ces.full_train(save_dir = modeldir+'ces', save_every = save_every)



#############################
# Apply conformal inference #
#############################

# Initialize result data frame
results = pd.DataFrame({})
#------------ Data splitting method ------------------#
print('Computing data splitting benchmark p-values for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

best_loss_bm, best_model_bm, test_val_loss_history_bm = CES_mc_bm.select_model()
model_list_bm = CES_mc_bm.model_list
C_PVals_bm = Conformal_PVals(net_bm, device, calib_loader_bm, model_list_bm, random_state = seed)

pvals_bm = C_PVals_bm.compute_pvals(inputs, [best_model_bm]*len(inputs))
results_bm = eval_pvalues(pvals_bm, is_outlier, alpha_list)
results_bm["Method"] = "Data Splitting"
results_bm["Alpha"] = alpha_list[0]
results = pd.concat([results, results_bm])


#------------ Naive method + Theory ------------------#
print('Computing naive benchmark p-values for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

model_list_ces = CES_mc_ces.model_list
C_PVals_ces = Conformal_PVals(net_ces, device, escal_loader_ces, model_list_ces, random_state = seed)

best_loss_naive, best_model_naive, val_loss_history_naive = CES_mc_ces.select_model()
pvals_naive = C_PVals_ces.compute_pvals(inputs, [best_model_naive]*len(inputs))
T=len(model_list_ces)

# results with theoretical correction
alpha_correct_list = list(map(theory.inv_hybrid, [T]*len(alpha_list), \
                              [n_escal_ces]*len(alpha_list), alpha_list))
unclipped_alpha = alpha_correct_list
alpha_correct_list = np.clip(alpha_correct_list, 1.0/n_escal_ces, 1)
results_theory = eval_pvalues(pvals_naive, is_outlier, alpha_correct_list)
results_theory["Method"] = "Theory"
results_theory["Alpha"] = [[unclipped_alpha[0],alpha_correct_list[0]]]
results = pd.concat([results, results_theory])

# results without theoretical correction
results_naive = eval_pvalues(pvals_naive, is_outlier, alpha_list)
results_naive["Method"] = "Naive"
results_naive["Alpha"] = alpha_list[0]
results = pd.concat([results, results_naive])


#------------ Full training ------------------#
print('Computing full training benchmark p-values for {:d} test points...'.format(n_test_samples))
full_model = model_list_ces[-1]
pvals_full = C_PVals_ces.compute_pvals(inputs, [full_model]*len(inputs))
results_full = eval_pvalues(pvals_full, is_outlier, alpha_list)
results_full["Method"] = "Full Training"
results_full["Alpha"] = alpha_list[0]
results = pd.concat([results, results_full])


#------------ CES ------------------#
print('Computing CES p-values for {:d} test points...'.format(n_test_samples))
best_loss_ces, best_model_ces, test_val_loss_history_ces = CES_mc_ces.select_model(inputs)
best_model_ces = list(np.array(best_model_ces)[:,3])
pvals_ces = C_PVals_ces.compute_pvals(inputs, best_model_ces)
results_ces = eval_pvalues(pvals_ces, is_outlier, alpha_list)
results_ces["Method"] = "CES"
results_ces["Alpha"] = alpha_list[0]
results = pd.concat([results, results_ces])


################
# Save Results #
################
results = add_header(results)
results.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir, ignore_errors=True)
