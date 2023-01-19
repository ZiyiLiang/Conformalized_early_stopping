import numpy as np
import sys, os
import pandas as pd
import torch as th
import torch.optim as optim
import shutil
import torchvision
from torchvision import transforms
from torchvision import datasets

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from third_party.datasetMaker import get_class_i, DatasetMaker
from ConformalizedES.method import CES_multiClass
from ConformalizedES.networks import SimpleConvolutionalNetwork
from ConformalizedES.inference import Conformal_PSet
from ConformalizedES.utils import eval_m_psets
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
alpha = 0.1



###############
# Output file #
###############
outdir = "results/multiclass/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_data) + "_lr" + str(lr) + "_epoch" + str(n_epoch) +\
                 "_seed" + str(seed)
outfile = outdir + outfile_name + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

modeldir = "models/multiClass/"+outfile_name+"/"

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
num_workers = 4

train_loader_bm = th.utils.data.DataLoader(train_set_bm, batch_size=batch_size,
                                          num_workers=num_workers)

es_loader_bm = th.utils.data.DataLoader(es_set_bm, batch_size=int(n_es_bm/5),
                                          num_workers=num_workers)

calib_loader_bm = th.utils.data.DataLoader(calib_set_bm, batch_size=int(n_calib_bm/5),
                                          num_workers=num_workers)

# For CES
train_loader_ces = th.utils.data.DataLoader(train_set_ces, batch_size=batch_size,
                                          num_workers=num_workers)

escal_loader_ces = th.utils.data.DataLoader(escal_set_ces, batch_size=int(n_escal_ces/5),
                                          num_workers=num_workers)


# Test loader
np.random.seed(seed)
th.manual_seed(seed)

n_test = len(test_set)

test_sample, _ = th.utils.data.random_split(test_set,[n_test_samples, n_test-n_test_samples])
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
# Create loss and optimizer
# CES_multiClass object will assume criterion takes three parameters: output, input and target, 
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
CES_mc_ces.full_train(save_dir =  modeldir+'ces', save_every = save_every)




#############################
# Apply conformal inference #
#############################

# Initialize result data frame
results = pd.DataFrame({})
#------------ Data splitting method ------------------#
print('Computing data splitting benchmark Psets for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

best_loss_bm, best_model_bm, test_val_loss_history_bm = CES_mc_bm.select_model()
model_list_bm = CES_mc_bm.model_list     # Get the saved model list from the CES class
C_PSet_bm = Conformal_PSet(net_bm, device, calib_loader_bm, n_classes, model_list_bm, \
                           alpha,lc=False, random_state = seed)

# Get the marginal conformal pvalues 
pset_m_bm= C_PSet_bm.pred_set(inputs, [[best_model_bm]*n_classes]*len(inputs), marginal=True)
results_bm = eval_m_psets(pset_m_bm, labels.numpy())
results_bm["Method"] = "Data Splitting"
results_bm["Alpha"] = alpha
results = pd.concat([results, results_bm])


#------------ Naive method + Theory ------------------#
print('Computing naive benchmark Psets for {:d} test points...'.format(n_test_samples))
sys.stdout.flush()

model_list_ces = CES_mc_ces.model_list 
C_PSet_ces = Conformal_PSet(net_ces, device, escal_loader_ces, n_classes, model_list_ces, \
                           alpha,lc=False,random_state = seed)

# results without theoretical correction
best_loss_naive, best_model_naive, test_val_loss_history_naive = CES_mc_ces.select_model()
pset_m_naive= C_PSet_ces.pred_set(inputs, [[best_model_naive]*n_classes]*len(inputs), marginal=True)
results_naive = eval_m_psets(pset_m_naive, labels.numpy())
results_naive["Method"] = "Naive"
results_naive["Alpha"] = alpha
results = pd.concat([results, results_naive])


# results with theoretical correction
T=len(model_list_ces)
alpha_correct = theory.inv_hybrid(T, n_escal_ces, alpha)
alpha_correct = np.clip(alpha_correct, 1.0/n_escal_ces, 1)
C_PSet_correct = Conformal_PSet(net_ces, device, escal_loader_ces, n_classes, model_list_ces, \
                           alpha_correct,lc=False,random_state = seed)
pset_m_theory= C_PSet_correct.pred_set(inputs, [[best_model_naive]*n_classes]*len(inputs), marginal=True)
results_theory = eval_m_psets(pset_m_theory, labels.numpy())
results_theory["Method"] = "Theory"
results_theory["Alpha"] = alpha_correct
results = pd.concat([results, results_theory])


#------------ Full training ------------------#
print('Computing full training benchmark Psets for {:d} test points...'.format(n_test_samples))
full_model = model_list_ces[-1]
pset_m_full= C_PSet_ces.pred_set(inputs, [[full_model]*n_classes]*len(inputs), marginal=True)
results_full = eval_m_psets(pset_m_full, labels.numpy())
results_full["Method"] = "Full Training"
results_full["Alpha"] = alpha
results = pd.concat([results, results_full])

#------------ CES ------------------#
print('Computing CES Psets for {:d} test points...'.format(n_test_samples))
best_loss_ces, best_model_ces, test_val_loss_history_ces = CES_mc_ces.select_model(inputs)
pset_m_ces= C_PSet_ces.pred_set(inputs, best_model_ces, marginal=True)
results_ces = eval_m_psets(pset_m_ces, labels.numpy())
results_ces["Method"] = "CES"
results_ces["Alpha"] = alpha
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
