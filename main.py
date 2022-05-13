

from mylsh import load_groundtruth, load_testdata, run_minhash_experiment, run_minhashLSH_experiment
from datasketch import LeanMinHash   

numDocs = 1000   

plagiaries = load_groundtruth(path="./MinHash/data", numdocs=numDocs)  
textdata = load_testdata(path="./MinHash/data", numdocs=numDocs)

run_minhash_experiment(textdata, plagiaries)
run_minhashLSH_experiment(textdata, plagiaries, num_hashes=100, 
                              n_grams=2, n_bands=10)
    





        
# All results: Recall 100%, Precision 100%


# Timing: BASELINE (GITHUB CODE). numHashes = 10
#               Total time       Shingles      Generating Minhash   Comparing Minhash   Finding most similar
# 1000 docs:    4.0sec total            0.4sec          1.5sec          1.7sec          0.4sec
# 2500 docs:    18.7sec total           1.0sec          3.7sec          11.2sec         2.9sec
# 10000 docs:  247.9sec total          3.9sec          19.6sec         182.2sec         42.1sec

# Timing: BASELINE (GITHUB CODE). numHashes = 100
# 1000 docs:    26.7sec total           0.4sec          14.3sec         11.5sec         0.4sec
# 2500 docs:    105.9sec total          1.0sec          32.9sec         69.5sec         2.6sec

# NUMPY/NUMBA/SCIPY implementation. MINHASH numHashes = 10 
#               Total time       Shingles      Generating Minhash   Comparing Minhash
# 1000 docs:    0.15sec total		0.03sec		0.13sec		       0.00052sec
# 10000 docs    5.92sec total	    2.43sec		 0.55sec		   2.9sec
# 10000 docs*NN 34.15sec total		2.80sec		28.31sec (!)		3.0sec
# *NN: No Numba for minhash_shingles

# NUMPY/NUMBA/SCIPY implementation. MINHASH + LSH numHashes = 100 
#               Total time       Shingles      Generating Minhash   Comparing Minhash
# 1000 docs:    0.77sec total		0.26sec		0.50sec		0.0029sec
# 10000 docs:   7.12sec total		2.50sec		4.59sec		0.031sec



