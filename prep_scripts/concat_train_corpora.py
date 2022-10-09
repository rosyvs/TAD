import pandas as pd
import os
import numpy as np
import csv

SRATE = 16000
MIN_DUR_S = 1 # filter to apply to corpora
APPLY_FILTER = False
VERSTR = '_fixlen1'

CORPORA_PATH = '/mnt/shared/CORPORA/'
csv_out = os.path.join(CORPORA_PATH, 'data_manifests', f'ALL_TRAIN{VERSTR}.csv')

manifest_dir = os.path.join(CORPORA_PATH, 'data_manifests')
concat_list = [
    f'ISAT-SI_TRAIN{VERSTR}.csv',
    f'myst_train{VERSTR}.csv',
    f'myst_test{VERSTR}.csv',
    f'myst_development{VERSTR}.csv',
    f'cukids_train-part1-cu{VERSTR}.csv',
    f'cukids_train-part2-cu-sentences{VERSTR}.csv',
    f'cukids_train-part3-cu-stories{VERSTR}.csv',
    f'cukids_train-part4-cu-summaries{VERSTR}.csv',
    f'cukids_train-part5-ogi-1-5{VERSTR}.csv',
    f'vox1_dev{VERSTR}.csv',
    f'cslu_scripted{VERSTR}.csv',
    f'cslu_spontaneous{VERSTR}.csv']





combined_csv = pd.concat(
    [pd.read_csv(os.path.join(manifest_dir, f)) for f in concat_list]).reset_index(drop=True)

# apply filter conditions
if APPLY_FILTER:
    b4=len(combined_csv)
    combined_csv = combined_csv[combined_csv['duration']>=SRATE*MIN_DUR_S]
    after=len(combined_csv)
    print(f'removed {b4-after} files not meeting minimum duration {MIN_DUR_S} sec')

# check for duplicate IDs and add an appendix if so
duplicates =combined_csv[combined_csv['ID'].duplicated()]
if len(duplicates)>0:
    print(f'duplicated IDs: {len(duplicates)}')
    IDcount=combined_csv.groupby('ID', as_index=False).cumcount()
    m = combined_csv['ID'].duplicated(keep=False) # Mask for all duplicated values

    # append duplicate count to ID to make unique
    combined_csv.loc[m, 'ID'] +=( "_" + IDcount[m].astype(str))

    duplicates =combined_csv[combined_csv['ID'].duplicated()]
    if len(duplicates)>0:
        print(f'STILL duplicated IDs: {len(duplicates)} WTF')

combined_csv.to_csv(csv_out,index=False)

# csv_file = ''
# # Writing the csv lines
# with open(csv_out, mode="w") as csv_f:
#     csv_writer = csv.writer(
#         csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
#     )
#     for line in combined_csv:
#         csv_writer.writerow(line)

# give some stats
durs = [row['duration'] for i, row in combined_csv.iterrows()]
spkrs = [row['speaker'] for i, row in combined_csv.iterrows()]

print(f'{len(combined_csv)} utterances, mean duration {np.mean(durs)/SRATE:.2f} sec, {len(set(spkrs))} speakers')
