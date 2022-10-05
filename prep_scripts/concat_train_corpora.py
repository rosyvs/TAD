import pandas as pd
import os
import numpy as np
import csv

SRATE = 16000

CORPORA_PATH = '/mnt/shared/CORPORA/'
csv_out = os.path.join(CORPORA_PATH,'data_manifests',f'ALL_TRAIN.csv')

manifest_dir = os.path.join(CORPORA_PATH,'data_manifests')
concat_list = ['cukids_train-part4-cu-summaries.csv',
 'cslu_spontaneous.csv',
 'myst_train.csv',
 'myst_development.csv',
 'cukids_train-part5-ogi-1-5.csv',
 'NOISE_ISAT-SI.csv',
 'cukids_train-part1-cu.csv',
 'cukids_train-part3-cu-stories.csv',
 'vox1_dev.csv',
 'cukids_train-part2-cu-sentences.csv',
 'cslu_scripted.csv',
 'ISAT-SI_TRAIN.csv',
 'myst_test.csv']



combined_csv = pd.concat([pd.read_csv(os.path.join(manifest_dir,f)) for f in concat_list ])
csv_file = ''
# Writing the csv lines
with open(csv_out, mode="w") as csv_f:
    csv_writer = csv.writer(
        csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    for line in combined_csv:
        csv_writer.writerow(line)

# give some stats
durs = [row['duration'] for i,row in combined_csv.iterrows()]
spkrs = [row['speaker'] for i,row in combined_csv.iterrows()]
durs=durs[1:]
spkrs = spkrs[1:]
print(f'{len(combined_csv)-1} utterances, mean duration {np.mean(durs)/SRATE:.2f} sec, {len(set(spkrs))} speakers')

