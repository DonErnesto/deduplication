# ======== MinHash =======
# E. Oldenhof. 
# Adaptation of code from https://github.com/chrisjmccormick/MinHash

# Design philosophy:
# - Should enable analysis of large document collections (so no storage of NxN matrices)
# - Use pd.Series to pass collection of texts with named indices
# - numpy vectorization / numba to speed-up operation (enabling a large number
#   of hash functions, say 100+, for LSH)



import time
import binascii

from collections import defaultdict

import pandas as pd
import numpy as np
#import numba
from numba import njit

    
from sklearn.metrics import pairwise_distances_chunked
# from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import murmurhash3_32

MAXSHINGLEHASH = 2**32-1 # we use 32-bit hashing --> 4.3E9 slots
PRIME = 4294967311


def _create_shingles_fromstring(text, n_grams=2, hashfunc=binascii.crc32):
    """
    Returns a set of shingles from a string. Tokenization through splitting by
    whitespaces (may be replaced by a tokenizer that can be passed in the future)
    
    text (string) : text to be shingled
    n_grams (int) : number of tokens in shingle (n in n-gram)
    hashfunc (None or a function) :  if None, return the raw shingles. 
                                    if a hash-function, returns its hashed representation
                                
    """
    shingles = set()
    words = text.split() #NB: split() deals with multiple spaces, tabs, etc.
    for index in range(0, len(words) - n_grams):
        shingle = ''.join(words[index : index + n_grams])
        if hashfunc:
            #Hash the byte-encoded shingle to a 32-bit integer
            shingle = hashfunc(shingle.encode(encoding='utf-8'))
        shingles.add(shingle)
    #crucial: return a np.array to profit from numba. makes 100x difference
    return np.array(list(shingles)) 



def shingles_from_series(series, **kwargs):
    return series.apply(lambda x: _create_shingles_fromstring(x, **kwargs))


@njit    
def hash_func(x, A, B, m):
    return (A * x + B) % m

@njit
def _minhash_fromshingles(shingles, A, B, m):
    """
    Returns the minhash from a list of hashed shingles, applying the hash function 
    (A*x + B) % m
    NB: may not be optimal, but considerably faster than murmurhash
    """
    return np.min(hash_func(shingles, A, B, m))



def minhash_signature_fromshingles(shinglesdata, numhashes=10):
    """
    Returns a (dense) numpy array, with each row i a minhash signature of document
    i of length numhashes (NB: should better return DataFrame?)
    
    shinglesdata (pd.Series) : a iteratible, each element a list of shingles
    NB: may need speeding up as well
    """
    signature_array = np.zeros((len(shinglesdata), numhashes))
       
    #coef_list = [np.random.randint(0, MAXSHINGLEHASH, 2) for i in range(numhashes)]
    for j in range(numhashes):
            A, B = np.random.randint(0, MAXSHINGLEHASH, 2)
            signature_array[:, j] = shinglesdata.apply(lambda shingles: 
                    _minhash_fromshingles(shingles, A, B, MAXSHINGLEHASH))
    return signature_array


         
def pairwise_slow_minhash_comparison(signature_array, sim_threshold=0.5, index_names=None):
    """
    Returns a dictionary with indices for which the similarity of the 
    minhash signatures exceed sim_threshold
    Note that the fraction of identical hashes (Hamming distance) is the 
    expectation of the Jaccard similarity of the original data
    
    The key is an index, its value a list of elements that were similar
    NB: returns integer indices
    
    signature_array (np.array) : N_documents x N_minhashes
    sim_threshold : 
    index_names (None of a Series/index): if not None, maps index to key
    """
    if not index_names is None:
        assert len(index_names) == len(signature_array), 'length index names should correspond with signature_array'
    N = signature_array.shape[0]
    similar_items = defaultdict(list)
    for i in range(N):
        base_arr = signature_array[i, :] 
        for j in range(i+1, N):
            similarity = np.mean(base_arr == signature_array[j, :])
            if similarity > sim_threshold:
                if not index_names is None:
                    i_key, j_key = index_names[i], index_names[j]
                else:
                    i_key, j_key = i, j
                similar_items[i_key].append(j_key)
                similar_items[j_key].append(i_key)
    return similar_items



def pairwise_minhash_comparison(signature_array, sim_threshold=0.5, 
                                     index_names=None):
    """
    Returns a dictionary with indices for which the similarity of the 
    minhash signatures exceed sim_threshold.
    Makes use of scikit-learn's chunked pairwise distance generator with Hamming metric
    for performance and memory efficiency.

    The key is an index, its value a list of elements that were similar
    NB: returns integer indices
    
    signature_array (np.array) : N_documents x N_minhashes
    sim_threshold : 
    index_names (None of a Series/index): if not None, maps index to key
    """    
    if not index_names is None:
        assert len(index_names) == len(signature_array), 'length index names should correspond with signature_array'
    similar_items = defaultdict(list)
    
    # iterator over distance matrix. NB: distance = 1-similarity
    dist_gen = pairwise_distances_chunked(signature_array, 
                                          metric='hamming')
    
    start_row = 0 #start of chunk
    for i, dist_row in enumerate(dist_gen):
        # dist_gen contains as many rows as possible, given memory constraints
        # NB: distance = 1 - similarity. So criterion: distance < 1 - sim_threshold
        # duplicates_idx = np.where(dist_row <= 1 - sim_threshold)[1]
        duplicates_i, duplicates_j = np.where(dist_row <= 1 - sim_threshold)
        for i_0, j in zip(duplicates_i, duplicates_j):
            i = i_0 + start_row
            if j > i:
                if not index_names is None:
                    i_key, j_key = index_names[i], index_names[j]
                else:
                    i_key, j_key = i, j
                similar_items[i_key].append(j_key)
                similar_items[j_key].append(i_key)  
        start_row += dist_row.shape[0]
    return similar_items


            
def pairwise_cosine_comparison(textdata, index_names=None, sim_threshold=0.5):
    """ Brute-force: cosine comparison of all pairs
    Baseline performance. 
    """
    if not index_names is None:
        assert len(index_names) == len(textdata), 'length index names should correspond with signature_array'
    similar_items = defaultdict(list)

    cv = CountVectorizer(lowercase=False, binary=True, 
                         ngram_range=[1, 2], max_features=20000)
    oh_matrix = cv.fit_transform(textdata)
    
    # row-by-row iterator over distance matrix
    dist_gen = pairwise_distances_chunked(oh_matrix, 
                                          metric='cosine',
                                          working_memory=0)
    for i, dist_row in enumerate(dist_gen):
        duplicates_idx = np.where(dist_row < (1 - sim_threshold))[1]
        for j in duplicates_idx:
            if j > i:
                if not index_names is None:
                    i_key, j_key = index_names[i], index_names[j]
                else:
                    i_key, j_key = i, j
                similar_items[i_key].append(j_key)
                similar_items[j_key].append(i_key)  
    return similar_items


def lsh_for_signaturematrix(signature_matrix, n_bands, index_names=None):
    """
    Locality-Sensitve hashing for near-duplicate detection
    
    A signature-matrix (N rows, D columns) has D different minhash values for
    N documents. By dividing the D values into n_bands bands, and testing for exact
    similarity within these bands, we can compare the documents in linear time
    """
    if not index_names is None:
        assert len(index_names) == signature_matrix.shape[0], 'length index names should correspond with signature_array'
    assert not signature_matrix.shape[1] % n_bands, 'the number of columns of the '\
        'signature matrix should be an integer multiple of the number of bands'
    
    duplicate_candidates = defaultdict(list)
    band_width = signature_matrix.shape[1] // n_bands
    
    for i in range(n_bands):
        signature_band = signature_matrix[:, i * band_width : (i+1) * band_width]
        signature_band = signature_band.sum(axis=1) #NB: some - I guess very small -risk of collisions here
        # signature_band = np.apply_along_axis(np.str, arr=signature_band, axis=0)
        u, c = np.unique(signature_band, return_counts=True)
        duplicate_vals = u[c > 1]
        for duplicate_val in duplicate_vals:
            dup_indices = np.where(signature_band==duplicate_val)[0]
            
            for i in dup_indices:
                for j in dup_indices:
                    if i != j:
                        if not index_names is None:
                            i_key, j_key = index_names[i], index_names[j]
                        else:
                            i_key, j_key = i, j
                        duplicate_candidates[i_key].append(j_key)
                        
    # duplicate_candidates has all candidates for each band
    # reduce to a unique list
    for k, v in duplicate_candidates.items():
        v = list(set(v))
        duplicate_candidates[k] = v
        
    return duplicate_candidates
    

def test_performance(plagiaries, similar_items):
    """
    Scans through similar_items, to see what fraction is contained in 
    plagiates
    To avoid double counting, only values in the list that > key are counted
    """
    tp, fp = 0, 0
    n_found_duplicates = 0
    for s_key, s_values in similar_items.items():
        duplicates = [s_value for s_value in s_values if
                      s_value > s_key]
        for duplicate in duplicates:
            n_found_duplicates += 1
            if plagiaries.get(s_key, None) == duplicate:
                tp += 1
            else:
                fp += 1
    # determine number of duplicate pairs ( in the general case, 
    # where > 2 docs may be duplicates)
    n_true_duplicates = 0
    for p_key, p_values in plagiaries.items():
        if isinstance(p_values, list):
            n_true_duplicates += len([p_value for p_value in p_values if
                          p_value > p_key])
        else:
            if p_values > p_key:
                n_true_duplicates += 1 
            
    print('Of {:d} true duplicate pairs, {:d} were correctly identified (recall: {:.1f})'.format(
            n_true_duplicates, tp, tp /n_true_duplicates))
    print('Of {:d} detected duplicate pairs, {:d} were correctly identified (precision: {:.1f})'.format(
            n_found_duplicates, tp, tp / (tp + fp)))
         


# Loading of groundtruth
def load_groundtruth(path="./data", numdocs=1000):
    """
    Returns a dictionary with all duplicates as (key:value)
    NB: this is a restrictive form, where each article may have only one 
    duplicate. 
    """
    assert str(numdocs) in ('100', '1000', '2500', '10000'), 'Only 100 '\
    ', 1000, 2500 and 10000 allowed for numdocs'
    truthFile = path + "/articles_" + str(numdocs) + ".truth"
    plagiaries = {}
    # Open the truth file.
    with open(truthFile, "rU") as f:
        # For each line of the files...
        for line in f:
            # Strip the newline character, if present.
            if line[-1] == '\n':
                line = line[0:-1]
            docs = line.split(" ")
            # Map the two documents to each other.
            plagiaries[docs[0]] = docs[1]
            plagiaries[docs[1]] = docs[0] 
    return plagiaries


def load_testdata(path="./data", numdocs=1000):
    """
    Returns a Series with docID as index
    """
    dataFile = path + "/articles_" + str(numdocs) + ".train"
    data_dict = {}
    with open(dataFile, "rU", encoding="utf-8") as f:
        for i in range(0, numdocs):
            # Read all of the words (they are all on one line) 
            words = f.readline().split(' ')
            
            # The Article ID is the first word on the line.
            docID = words[0]
            data_dict[docID] = ' '.join(words[1:])
    return pd.Series(data_dict)


def run_minhash_experiment(textdata, plagiaries, num_hashes=10, n_grams=2):
    time_dict={}
    print('Running MinHASH analysis for {} documents, with {} hashes'.format(len(textdata), num_hashes))   
    
    # 1) Create shingles
    t0 = time.perf_counter()
    # shinglesdata = shingles_from_series(textdata, n_grams=n_grams)
    shinglesdata = shingles_from_series(textdata, n_grams=n_grams, hashfunc=murmurhash3_32)
    time_dict['Shingling text'] = time.perf_counter() - t0
    print('\nAverage number of shingles per document: {:.1f}'.format(
            shinglesdata.apply(lambda x: len(x)).mean()))

    # 2) Create signature matrix
    t1 = time.perf_counter()        
    signature_matrix = minhash_signature_fromshingles(shinglesdata, 
                                                      numhashes=num_hashes)
    time_dict['Generating Signature (minHash) matrix '] = time.perf_counter() - t1


    t3 = time.perf_counter()        
    # 3) Find similar pairs (pairwise, or N-choose-2 complexity)
    similar_items = pairwise_minhash_comparison(signature_matrix, sim_threshold=0.8,
                                                index_names=textdata.index)
    time_dict['Pairwise comparison'] = time.perf_counter()-t3
   
    time_dict['Total time'] = time.perf_counter()  - t0
    test_performance(plagiaries, similar_items)
    
    print('Total time       Shingles      Generating Minhash   Comparing Minhash')
    print('{:.2f}sec total\t\t{:.2f}sec\t\t{:.2f}sec\t\t{:.2}sec'.format(
            time_dict['Total time'],
            time_dict['Shingling text'],
            time_dict['Generating Signature (minHash) matrix '] ,
            time_dict['Pairwise comparison'] ))


def run_minhashLSH_experiment(textdata, plagiaries, num_hashes=20, 
                              n_grams=2, n_bands=5):
    time_dict={}
    print('\n\nRunning MinHASH+LSH analysis for {} documents, with {} hashes'.format(len(textdata), num_hashes))   

    # 1) Create shingles
    t0 = time.perf_counter()
    shinglesdata = shingles_from_series(textdata, n_grams=n_grams)
    time_dict['Shingling text'] = time.perf_counter() - t0
    print('\nAverage number of shingles per document: {:.1f}'.format(
            shinglesdata.apply(lambda x: len(x)).mean()))
    
    # 2) Create signature matrix        
    t1 = time.perf_counter()
    signature_matrix = minhash_signature_fromshingles(shinglesdata, 
                                                      numhashes=num_hashes)
    time_dict['Generating Signature (minHash) matrix '] = time.perf_counter() - t1


    # 3) Find identical chunks within the signature matrix
    t3 = time.perf_counter()          
    similar_items = lsh_for_signaturematrix(signature_matrix, n_bands=n_bands, 
                                            index_names=textdata.index)
    time_dict['LSH step'] = time.perf_counter()-t3
   
    time_dict['Total time'] = time.perf_counter()  - t0
    test_performance(plagiaries, similar_items)
    
    print('Total time       Shingles      Generating Minhash   LSH')
    print('{:.2f}sec total\t\t{:.2f}sec\t\t{:.2f}sec\t\t{:.2}sec'.format(
            time_dict['Total time'],
            time_dict['Shingling text'],
            time_dict['Generating Signature (minHash) matrix '] ,
            time_dict['LSH step'] ))

 
    
if __name__ == '__main__':


    
        
    numDocs = 1000
    
    plagiaries = load_groundtruth(path="./MinHash/data", numdocs=numDocs)  
    textdata = load_testdata(path="./MinHash/data", numdocs=numDocs)
    
    run_minhash_experiment(textdata, plagiaries)
    run_minhashLSH_experiment(textdata, plagiaries, num_hashes=100, 
                                  n_grams=2, n_bands=10)
    
    

# All results: Recall 100%, Precision >90%


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
# 100 docs:    0.15sec total		0.03sec		0.13sec		       0.00052sec
# 1000 docs:   0.50sec total		0.27sec		0.21sec		        0.018sec
# 10000 docs    5.92sec total	    2.43sec		 0.55sec		   2.9sec
# 10000 docs*NN 34.15sec total		2.80sec		28.31sec (!)		3.0sec
# *NN: No Numba for minhash_shingles

# NUMPY/NUMBA/SCIPY implementation. MINHASH + LSH numHashes = 100 
#               Total time       Shingles      Generating Minhash   Comparing Minhash
# 100 docs:    0.77sec total		0.26sec		0.50sec		    0.0029sec
# 1000 docs:   0.74sec total		0.25sec		0.49sec		    0.003sec
# 10000 docs:   7.12sec total		2.50sec		4.59sec		    0.031sec (indeed: Linear!!)
    