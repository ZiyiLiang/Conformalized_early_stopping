# CES (Conformalized Early Stopping)
This repository implements *Conformalized Early Stopping*: a novel method that combines early stopping with conformal calibration using only one set of hold-out data while convetional methods require additional data-splitting or conservative adjustments to meet the same theoretical guarantees. 


## Contents

 - `ConformalizedES/` Python package implementing our methods and some alternative benchmarks.
 - `third_party/` Third-party Python packages imported by our package.
 - `experiments/` Codes to replicate the figures and tables for the experiments with real data discussed in the accompanying paper.
  
    
## Prerequisites

Prerequisites for the CES package:
 - numpy
 - scipy
 - sklearn
 - skgarden
 - torch
 - tqdm
 - sympy
 - torchmetrics
 - collections
 - numdifftools
 - xml
 - math
 - pandas
 - matplotlib
 - statsmodels

Additional prerequisites to run the numerical experiments:
 - shutil
 - torchvision
 - tempfile
