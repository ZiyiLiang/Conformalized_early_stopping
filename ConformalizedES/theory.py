import numpy as np
import math
from math import sqrt,log, exp
from scipy.stats import beta

import pdb

def DKW_bound(T,n,a):
    """
    Parameters
    ----------
    T: int
        Number of models
    n: int
        Calibration size
    a: float
        Confidence level
    """
    # Calculate the error term
    err = sqrt(log(2*T)/2) + sqrt(2)*T*exp(-log(2*T))/(sqrt(log(2*T))+sqrt(log(2*T)+4/math.pi))
    bound = (1+1/n)*(1-a) - err/sqrt(n)
    return bound

def Markov_bound(T,n,a,b=100):
    l = int(a*(n+1))
    return (1-1/b)*beta.ppf(1/(T*b), n+1-l,l)

def hybrid_bound(T,n,a,b=100):
    err = sqrt(log(2*T)/2) + sqrt(2)*T*exp(-log(2*T))/(sqrt(log(2*T))+sqrt(log(2*T)+4/math.pi))
    dkw_bound = (1+1/n)*(1-a) - err/sqrt(n)
    
    l = int(a*(n+1))
    markov_bound = (1-1/b)*beta.ppf(1/(T*b), n+1-l,l)
    return max(markov_bound, dkw_bound)

def inv_DKW(T,n,a):
    """
    calculate the corrected confidence level when theoretical coverage rate is 1-a
    """
    err = sqrt(log(2*T)/2) + sqrt(2)*T*exp(-log(2*T))/(sqrt(log(2*T))+sqrt(log(2*T)+4/math.pi))
    err /= sqrt(n)
    ac = 1-(1-a+err)/(1+1/n)
    return ac if ac>0 else 0

def inv_Markov(T,n,a,b=100):
    # The lower bound is a stepwise function in terms of alpha when n is fixed 
    # Calculate all the distinct lower bound values
    aseq = np.arange(1/n,1,1/n)
    lseq = np.array([int(a*(n+1)) for a in aseq])
    bound = (1-1/b)*beta.ppf(1/(T*b), n+1-lseq,lseq)
    
    # Filter out the valid coverage
    valid_idx = np.where(bound>(1-a))[0]
    return 0 if not len(valid_idx) else aseq[valid_idx[bound[valid_idx].argmin()]]

def inv_hybrid(T,n,a,b=100):
    # The Markov correction
    aseq = np.arange(1/n,1,1/n)
    lseq = np.array([int(a*(n+1)) for a in aseq])
    bound = (1-1/b)*beta.ppf(1/(T*b), n+1-lseq,lseq)
    
    # Filter out the valid coverage
    valid_idx = np.where(bound>(1-a))[0]
    M_ac = 0 if not len(valid_idx) else aseq[valid_idx[bound[valid_idx].argmin()]]
    
    
    # The DKW correction
    err = sqrt(log(2*T)/2) + sqrt(2)*T*exp(-log(2*T))/(sqrt(log(2*T))+sqrt(log(2*T)+4/math.pi))
    err /= sqrt(n)
    D_ac = 1-(1-a+err)/(1+1/n)
    
    return max(D_ac, M_ac)
