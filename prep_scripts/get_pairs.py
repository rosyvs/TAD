from cgi import test
import os
import pandas as pd
import numpy as np
import random

################
#
# Generate test pairs for speaker verification 
# - balance match and mismatch
# - subsample all possible pairings to limit n pairs
# 
###############

    
# options
opts = {}
opts['density'] = 1.0 # float. Proportion of samples to use. If <1.0 will randomly sample from density*n pairings
opts['verstr'] = 'ISAT-SI_TEST' # will save the configuration with this filename

CORPORA_PATH = '/mnt/shared/CORPORA/ISAT-SI/'
METADATA_csv = f'{CORPORA_PATH}/TEST/METADATA.csv' # a single csv containing all utterances in dataset, with spaeker labels, paths, transcripts, etc
# takes:
# METADATA csv 
# opts

# returns: pairs csv


# generates configuration files for speaker verification tests, given a label file for each session
print('configuring speaker recognition test pairs with provided options:')
test_pairs = [] # for storing list of test comparisons and timestamps

METADATA = pd.read_csv(METADATA_csv)
metadata = METADATA[['filepath','duration_sec','speaker']]# get 4 cols needed for etst pairs
metadata['start_sec'] = 0.0 # because segments are already extracted as singe wavs we don't need to read from orig full file
metadata  = metadata[['filepath','start_sec','duration_sec','speaker']]
metadata['filepath'] = CORPORA_PATH+  metadata['filepath'].astype(str)
metadata.rename(columns={'duration_sec':'end_s', 'start_sec':'start_s','filepath':'path'}, inplace=True)
# pair each smaple with one from the same speaker and one from another speaker
for i,s1 in metadata.iterrows():
    # select same
    m = metadata['speaker'] == s1['speaker']
    matching = m[m].index
    nonmatching = m[~m].index
    same = metadata.iloc[matching[random.randrange(0,len(matching))]]
    other = metadata.iloc[nonmatching[random.randrange(0,len(nonmatching))]]

    row1 = pd.concat([s1.add_prefix('x1'), same.add_prefix('x2')])
    row1['match'] = True
    row0 = pd.concat([s1.add_prefix('x1'), other.add_prefix('x2')])
    row0['match'] = False
    test_pairs.append(row1)
    test_pairs.append(row0)


# prune test list
if not (float(opts['density']) ==1.0):
    test_pairs = random.sample(test_pairs, int(len(test_pairs)*opts['density']))

test_pairs = pd.DataFrame(test_pairs).reset_index(drop=True, inplace=False)

print(f'Generated speaker verification test config for {opts["verstr"]}')
print(f'...containing {len(test_pairs)} comparisons')
print(f'...of which {test_pairs["match"].sum()} are matched speakers')


# save test list
os.makedirs(os.path.join(CORPORA_PATH,'configs','speaker_verification'),exist_ok=True)
test_pairs.to_csv(os.path.join(CORPORA_PATH,'configs','speaker_verification', f'{opts["verstr"]}_testpairs.csv'),index=False)



