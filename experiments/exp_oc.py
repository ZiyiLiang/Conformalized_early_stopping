import numpy as np
import sys, os
import pandas as pd
import torch as th
import torch.optim as optim
import shutil
from torchvision import transforms
from torchvision import datasets

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from third_party.datasetMaker import get_class_i, DatasetMaker
from ConformalizedES.method import CES_oneClass
from ConformalizedES.networks import ConvAutoencoder
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
    n_data = 300
    lr = 0.1
    n_epoch = 200
    seed = 0

# Fixed experiment parameters
batch_size = 10           # batch size for training set
n_test_samples = 100
save_every = 1         # Save every training snapshot  
alpha_list = [0.1]



###############
# Output file #
###############
outdir = "results/oneclass/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_data) + "_lr" + str(lr) + "_epoch" + str(n_epoch) +\
                 "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/oneClass/"+outfile_name+"/"

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

# Download the MNIST Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=0.5, std=0.5)])

train_set_full = datasets.MNIST(root = "./data", train = True, download = True, transform=transform)
test_set_full = datasets.MNIST(root = "./data", train = False, download = True, transform=transform)

x_train_full = train_set_full.data
y_train_full = train_set_full.targets
x_test_full = test_set_full.data
y_test_full = test_set_full.targets

# Train set composed only of number 0
train_set = \
    DatasetMaker(
        [get_class_i(x_train_full, y_train_full, 0)]
    )

# Test set is a mixture of number 0 and 8
test_set = \
    DatasetMaker(
        [get_class_i(x_test_full, y_test_full, 0),
        get_class_i(x_test_full, y_test_full, 8),
]
    )

print('total number of available training data is: {:d}.'.format(len(train_set)))
print('total number of test data is {:d} in which {:d} are label 0 test data, {:d} are label 8 test data.'\
      .format(len(test_set), test_set.lengths[0],test_set.lengths[1]))
sys.stdout.flush()



######################
# Create Dataloader  #
######################

# Split the data
np.random.seed(seed)
th.manual_seed(seed)

# Benchmark data splitting: split the data into 3 sets
n_full = len(train_set)
n_es_bm = int(n_data*0.25)
n_calib_bm = n_es_bm
n_train_bm = n_data-n_calib_bm-n_es_bm


train_set_bm, es_set_bm, calib_set_bm, _ = th.utils.data.random_split(train_set,\
                                 [n_train_bm, n_es_bm, n_calib_bm, n_full-n_data])

print("Use {:d} training samples, {:d} early stopping samples, {:d} calibration samples for the Data splitting method."\
     .format(n_train_bm, n_es_bm, n_calib_bm))
sys.stdout.flush()

# CES data splitting: calibration set is not needed, merge back to the training set
n_train_ces = n_data-n_es_bm
n_escal_ces = n_es_bm

train_set_ces, escal_set_ces, _ = th.utils.data.random_split(train_set,\
                                 [n_train_ces, n_escal_ces, n_full-n_data])
print("Use {:d} training samples, {:d} early stopping samples, {:d} calibration samples for the CES method."\
     .format(n_train_ces, n_escal_ces, n_escal_ces))
sys.stdout.flush()


# Create data loader objects
# For benchmarks
num_workers = 0

train_loader_bm = th.utils.data.DataLoader(train_set_bm, batch_size=batch_size,
                                          num_workers=num_workers)

es_loader_bm = th.utils.data.DataLoader(es_set_bm, batch_size=n_es_bm,
                                          num_workers=num_workers)

calib_loader_bm = th.utils.data.DataLoader(calib_set_bm, batch_size=n_calib_bm,
                                          num_workers=num_workers)

# For CES
train_loader_ces = th.utils.data.DataLoader(train_set_ces, batch_size=batch_size,
                                          num_workers=num_workers)

escal_loader_ces = th.utils.data.DataLoader(escal_set_ces, batch_size=n_escal_ces,
                                          num_workers=num_workers)


# Test loader
np.random.seed(seed)
th.manual_seed(seed)

n_test = len(test_set)

test_sample, more_sample = th.utils.data.random_split(test_set,[n_test_samples, n_test-n_test_samples])
test_loader = th.utils.data.DataLoader(test_sample, batch_size=n_test_samples, num_workers=num_workers)

# get all test images
dataiter = iter(test_loader)
inputs, labels = dataiter.next()




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
# CES_oneClass objs and opect will assume criterion takes three parameters: output, input and target, 
# create wrapper function to modify the criterion.
net_bm = ConvAutoencoder()
Loss = th.nn.MSELoss()
def criterion(outputs, inputs, targets):
    return Loss(outputs, inputs)
optimizer_bm = optim.Adam(net_bm.parameters(), lr=lr)

CES_oc_bm = CES_oneClass(net_bm, device, train_loader_bm, batch_size=batch_size, max_epoch=n_epoch, 
                        learning_rate=lr, val_loader=es_loader_bm, criterion=criterion,optimizer=optimizer_bm)

# Train the model and save snapshots regularly
CES_oc_bm.full_train(save_dir = modeldir+'benchmarks', save_every = save_every)


#------------ Training with without data splitting ------------------#
net_ces = ConvAutoencoder()
Loss = th.nn.MSELoss()
def criterion(outputs, inputs, targets):
    return Loss(outputs, inputs)
optimizer_ces = optim.Adam(net_ces.parameters(), lr=lr)


# Initialize the CES class with model parameters
CES_oc_ces = CES_oneClass(net_ces, device, train_loader_ces, batch_size=batch_size, max_epoch=n_epoch, 
                        learning_rate=lr, val_loader=escal_loader_ces, criterion=criterion,optimizer=optimizer_ces)

# Train the model and save snapshots regularly
CES_oc_ces.full_train(save_dir = modeldir+'ces', save_every = save_every)



#############################
# Apply conformal inference #
#############################

# Initialize result data frame
results = pd.DataFrame({})
#------------ Data splitting method ------------------#
print('Computing data splitting benchmark p-values for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

best_loss_bm, best_model_bm, test_val_loss_history_bm = CES_oc_bm.select_model()
model_list_bm = CES_oc_bm.model_list
C_PVals_bm = Conformal_PVals(net_bm, device, calib_loader_bm, model_list_bm, random_state = seed)

pvals_bm = C_PVals_bm.compute_pvals(inputs, [best_model_bm]*len(inputs))
results_bm = eval_pvalues(pvals_bm, labels, alpha_list)
results_bm["Method"] = "Data Splitting"
results = pd.concat([results, results_bm])


#------------ Naive method + Theory ------------------#
print('Computing naive benchmark p-values for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

model_list_ces = CES_oc_ces.model_list
C_PVals_ces = Conformal_PVals(net_ces, device, escal_loader_ces, model_list_ces, random_state = seed)

best_loss_naive, best_model_naive, val_loss_history_naive = CES_oc_ces.select_model()
pvals_naive = C_PVals_ces.compute_pvals(inputs, [best_model_naive]*len(inputs))
T=len(model_list_ces)

# results with theoretical correction
alpha_correct_list = list(map(theory.inv_hybrid, [T]*len(alpha_list), \
                              [n_escal_ces]*len(alpha_list), alpha_list))
alpha_correct_list = np.clip(alpha_correct_list, 1.0/n_escal_ces, 1)
results_theory = eval_pvalues(pvals_naive, labels, alpha_correct_list)
results_theory["Method"] = "Theory"
results = pd.concat([results, results_theory])

# results without theoretical correction
results_naive = eval_pvalues(pvals_naive, labels, alpha_list)
results_naive["Method"] = "Naive"
results = pd.concat([results, results_naive])


#------------ Full training ------------------#
print('Computing full training benchmark p-values for {:d} test points...'.format(n_test_samples))
full_model = model_list_ces[-1]
pvals_full = C_PVals_ces.compute_pvals(inputs, [full_model]*len(inputs))
results_full = eval_pvalues(pvals_full, labels, alpha_list)
results_full["Method"] = "Full Training"
results = pd.concat([results, results_full])


#------------ CES ------------------#
print('Computing CES p-values for {:d} test points...'.format(n_test_samples))
best_loss_ces, best_model_ces, test_val_loss_history_ces = CES_oc_ces.select_model(inputs)
pvals_ces = C_PVals_ces.compute_pvals(inputs, best_model_ces)
results_ces = eval_pvalues(pvals_ces, labels, alpha_list)
results_ces["Method"] = "CES"
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
