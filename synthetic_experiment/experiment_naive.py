import numpy as np
import sys, os
import pandas as pd
import torch as th


sys.path.append("../ConformalizedES")
sys.path.append("../synthetic_experiment")
sys.path.append("../third_party")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from toy_dataset import Blob_dataset, load_test, load_train, load_val_cal
from toy_network import SimpleNN
from experiment_utils import plot_loss_acc
from method import CES_oneClass
from utils import eval_pvalues
import torch.optim as optim
from inference import Conformal_PVals

#########################
# Experiment parameters #
#########################

if True: # Input parameters
    # Parse input arguments
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if len(sys.argv) != 8:
        print("Error: incorrect number of parameters.")
        quit()

    n_cal = int(sys.argv[1])
    lr = float(sys.argv[2])
    n_epoch = int(sys.argv[3])
    optimizer = sys.argv[4]
    random_state = int(sys.argv[5])
    std = float(sys.argv[6])
    n_features = int(sys.argv[7])


else: # Default parameters
    n_cal = 20
    lr = 1.0
    n_epoch = 50
    optimizer = "SGD"
    random_state = 0
    std = 2
    n_features = 2

# Define the model parameters


# Fixed experiment parameters
n_train = 100
n_val = n_cal
batch_size = 10 
n_test = 1000
alpha_list = [0.1]
n_classes = 2
num_repetitions = 2


###############
# Output file #
###############
outdir = "results/naive"
# os.makedirs(outdir, exist_ok=True)
outfile_prefix = outdir + "/" + "ncal"+str(n_cal) + "_lr" + str(lr) + "_epoch" + str(n_epoch) +\
                 "_optim" + str(optimizer) + "_seed" + str(random_state) + "_std" + str(std) + "_nfeature" + str(n_features)
outfile = outfile_prefix + ".txt"
print("Output file: {:s}".format(outfile), end="\n")

# Header for results file
def add_header(df):
    df["n_epoch"] = n_epoch
    df["n_calib"] = n_cals
    df["seed"] = random_state
    return df
    
###################
# Toy dataset #
###################

# center and standard deviation of the blobs
centers = np.array([[0,0],[3,3]])
null_center =  np.array([[0,0]])




###################
# Run experiments #
###################

def run_experiment(seed, train_set, n_train, val_set, n_val, n_cal, test_set, n_test,
                   lr, n_epoch, alpha_list, batch_size=10, num_worker=0, visualize=False):
    # Initialize result data frame
    results = pd.DataFrame({})
    
    # Get the dataloaders and test points
    train_loader_bm, train_loader_ces = load_train(seed, train_set, n_train, batch_size, num_worker),\
                                        load_train(seed, train_set, n_train, batch_size, num_worker)
    val_loader_bm, cal_loader_bm = load_val_cal(seed, val_set, n_val, n_cal, batch_size, num_worker)
    val_loader_ces, _ = load_val_cal(seed, val_set, n_val, n_cal, batch_size, num_worker)

    inputs, labels = load_test(seed, test_set, n_test)
    
    
    # Define default device, we should use the GPU (cuda) if available
    device = th.device("cuda" if th.cuda.is_available() else "cpu")### Define subset of the dataset (so it is faster to train)
    if th.cuda.is_available():
        # Make CuDNN Determinist
        th.backends.cudnn.deterministic = True
        th.cuda.manual_seed(seed)
#     Set the NN parameters 
    net_bm = SimpleNN(in_shape=inputs.shape[1], out_shape=2)
    Loss = th.nn.CrossEntropyLoss()
    def criterion(outputs, inputs, targets):
        return Loss(outputs, targets)
    
    if optimizer == "SGD":
        optimizer_bm = optim.SGD(net_bm.parameters(), lr=lr)
    else:
        optimizer_bm = optim.Adam(net_bm.parameters(), lr=lr)

    
    np.random.seed(seed)
    th.manual_seed(seed)
    
    # Train with benchmark data splitting
    print("Training with standard data splitting...")
    sys.stdout.flush()
    
    CES_oc_bm = CES_oneClass(net_bm, device, train_loader_bm, batch_size=batch_size, max_epoch=n_epoch, 
                        learning_rate=lr, val_loader=val_loader_bm, criterion=criterion,optimizer=optimizer_bm)
    CES_oc_bm.full_train(save_dir = './models/oneClass/exp'+str(seed)+'/benchmarks/', save_every = 1)
    
    if visualize:
        plot_loss_acc(CES_oc_bm.train_loss_history, CES_oc_bm.val_loss_history, 
                      CES_oc_bm.train_acc_history, CES_oc_bm.val_acc_history)
    
    # Compute the benchmark p-values
    print('Computing standard benchmark p-values for {:d} test points...'.format(n_test))
    sys.stdout.flush()
    
    best_loss_bm, best_model_bm, val_loss_history_bm = CES_oc_bm.select_model()
    model_list_bm = CES_oc_bm.model_list
    C_PVals_bm = Conformal_PVals(net_bm, device, cal_loader_bm, model_list_bm, random_state = seed)
    pvals_bm = C_PVals_bm.compute_pvals(inputs, [best_model_bm]*len(inputs))
    results_bm = eval_pvalues(pvals_bm, labels, alpha_list)
    results_bm["Method"] = "Standard benchmark"
    results_bm["train_loss_history"] = [CES_oc_bm.train_loss_history]
    results_bm["val_loss_history"] = [CES_oc_bm.val_loss_history]
    results_bm["train_acc_history"] = [CES_oc_bm.train_acc_history]
    results_bm["val_acc_history"] = [CES_oc_bm.val_acc_history]
    results = pd.concat([results, results_bm])
    
    
    np.random.seed(seed)
    th.manual_seed(seed)

    net_ces = SimpleNN(in_shape=inputs.shape[1], out_shape=2)
#     optimizer_ces = optim.Adam(net_ces.parameters(), lr=lr)
    optimizer_ces = optim.SGD(net_ces.parameters(), lr=lr)
    if optimizer == "SGD":
        optimizer_ces = optim.SGD(net_ces.parameters(), lr=lr)
    else:
        optimizer_ces = optim.Adam(net_ces.parameters(), lr=lr)
    
    # Initialize the CES class with model parameters
    print("Training with CES data splitting...")
    sys.stdout.flush()
    
    CES_oc_ces = CES_oneClass(net_ces, device, train_loader_ces, batch_size=batch_size, max_epoch=n_epoch, 
                            learning_rate=lr, val_loader=val_loader_ces, criterion=criterion,optimizer=optimizer_ces)
    CES_oc_ces.full_train(save_dir = './models/oneClass/exp'+str(seed)+'/ces/', save_every = 1)
    
    if visualize:
        plot_loss_acc(CES_oc_ces.train_loss_history, CES_oc_ces.val_loss_history, 
                      CES_oc_ces.train_acc_history, CES_oc_ces.val_acc_history)
    
    model_list_ces = CES_oc_ces.model_list
    C_PVals_ces = Conformal_PVals(net_ces, device, val_loader_ces, model_list_ces, random_state = seed)
    
    # Compute the benchmark p-values
    print('Computing naive benchmark p-values for {:d} test points...'.format(n_test))
    best_loss_naive, best_model_naive, val_loss_history_naive = CES_oc_ces.select_model()
    pvals_naive = C_PVals_ces.compute_pvals(inputs, [best_model_naive]*len(inputs))
    results_naive = eval_pvalues(pvals_naive, labels, alpha_list)
    results_naive["Method"] = "Naive benchmark"
    results_naive["train_loss_history"] = [CES_oc_ces.train_loss_history]
    results_naive["val_loss_history"] = [CES_oc_ces.val_loss_history]
    results_naive["train_acc_history"] = [CES_oc_ces.train_acc_history]
    results_naive["val_acc_history"] = [CES_oc_ces.val_acc_history]
    results = pd.concat([results, results_naive])
    
    return results


# Initialize result data frame
results = pd.DataFrame({})


for r in range(num_repetitions):
    print("\nStarting repetition {:d} of {:d}:\n".format(r+1, num_repetitions))
    sys.stdout.flush()
    
    # Change random seed for this repetition
    seed = 10*random_state*num_repetitions + r

    # Generate toy dataset
    train_set = Blob_dataset(5000, centers, n_features, std, random_state= seed)
    val_set =  Blob_dataset(5000, null_center, n_features, std, random_state=seed)
    test_set = Blob_dataset(5000, centers, n_features, std, random_state=seed)
    results_new = run_experiment(seed, train_set, n_train, val_set, n_val, n_cal, 
                                 test_set, n_test, lr, n_epoch, alpha_list, batch_size)
    results_new = add_header(results_new)
    results_new["Repetition"] = r
    results = pd.concat([results, results_new])
    # Save results
    results.to_csv(outfile, index=False)
    print("\nResults written to {:s}\n".format(outfile))
    sys.stdout.flush()

print("\nAll experiments completed.\n")
sys.stdout.flush()