import torch as th
import numpy as np

from torch.utils.data import Dataset
from sklearn.datasets import make_blobs
from torch.utils.data.sampler import SubsetRandomSampler



class Blob_dataset(Dataset):
    def __init__(self,n_samples, centers, n_features, std, random_state=0):
        self.X, self.y = make_blobs(n_samples=n_samples, centers=centers, 
                          n_features=n_features, cluster_std=std, random_state=random_state)
        self.X = self.X.astype('float32')
        self.y = self.y.astype('int64')
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def load_train(seed, train_set, n_train, batch_size=10, num_workers=0):
    np.random.seed(seed)
    th.manual_seed(seed)
    n_full = len(train_set)

    train_set, _ = th.utils.data.random_split(train_set, [n_train, n_full-n_train])
    train_loader = th.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          num_workers=num_workers)
    return train_loader


def load_test(seed, test_set, n_test, num_workers=0):
    np.random.seed(seed)
    th.manual_seed(seed)

    test_sampler = SubsetRandomSampler(np.arange(n_test, dtype=np.int64))
    test_loader = th.utils.data.DataLoader(test_set, batch_size=n_test, sampler=test_sampler,
                                             num_workers=num_workers)

    # get all test images
    dataiter = iter(test_loader)
    inputs, labels = dataiter.next()
    return inputs, labels

def load_val_cal(seed, val_set, n_val, n_cal, batch_size=10, num_workers=0):
    np.random.seed(seed)
    th.manual_seed(seed)
    
    n_full = len(val_set)
    n_data = n_val+n_cal

    val_set, cal_set, _ = th.utils.data.random_split(val_set, [n_val, n_cal, n_full-n_data])
    cal_loader = th.utils.data.DataLoader(cal_set, batch_size=batch_size,
                                             num_workers=num_workers)
    val_loader = th.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             num_workers=num_workers)

    return val_loader, cal_loader