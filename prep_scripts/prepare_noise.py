import os
import csv
from tqdm import tqdm
from prep_utils import split_to_chunks
import numpy as np
import pandas as pd
import math
import glob
import contextlib
import wave

CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'ISAT-NOISE/NOISE_wav'
SRATE = 16000
append_MUSAN = False # apend the existing csv for MUSAN
MUSAN_csv = os.path.join(CORPORA_PATH,'data_manifests', 'musan_noise.csv')


# prepare noise: ISAT noise + concatenate its csv to MUSAN noise

# for EnvCorrupt, noise.csv has lines which look like the following:
# ID,duration,wav,wav_format,wav_opts 
# where duration is in sec. wav is wav_file. wav_format is just 'wav'. wav_opts is TODO

# noise-free-sound-0842_99,3.0,/mnt/shared/CORPORA//augment/RIRS_NOISES/pointsource_noises/noise-free-sound-0842_99.wav,wav,


wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'*.wav'), recursive=True)
print(f'detected {len(wav_lst)} .wav files. Collecting metadata for csv...')

#TODO: what should wav_opts be? musan has this as blank
csv_output = [['ID','duration','wav','wav_format','wav_opts' ]]
csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'isat{ "+musan" if append_MUSAN else ""}_noise.csv')

for wav_file in tqdm(wav_lst):
    # Getting sentence and speaker ids
    try:
        recordingID = wav_file.split("/")[-1]
        ID = 'ISAT-SI_noise_' + recordingID.replace('.wav','')
        print(ID)
    except:
        print(f"Malformed path: {wav_file}")
        continue
    with contextlib.closing(wave.open(wav_file,'r')) as f:
        srate = f.getframerate()
        duration = f.getnframes()
        duration_sec = f.getnframes()/srate
    
    if srate != SRATE:
        raise ValueError(f'sampling rate is {srate}, but {SRATE} is required. Go back and reformat this corpus.')
    # append to csv rows
    csv_line = [
        ID,
        duration_sec,
        wav_file,
        'wav',# dummy entry
        ' '# dummy entry
    ]
    csv_output.append(csv_line)



musan_csv_output = [['ID','duration','wav','wav_format','wav_opts' ]]
with open(MUSAN_csv) as f:
    csv_reader = csv.reader(f, delimiter=',')
    next(csv_reader)
    next(csv_reader)
    for row in csv_reader:
        [ID,duration,wav,wav_format,wav_opts]= row
        row_fixed = [ID,duration,wav,wav_format,' ' if not wav_opts.strip() else wav_opts]
        musan_csv_output.append(row_fixed)
        if append_MUSAN:
            csv_output.append(row_fixed)

# Writing the csv lines
with open(MUSAN_csv, mode="w") as csv_f:
    csv_writer = csv.writer(
        csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    for line in musan_csv_output:
        csv_writer.writerow(line)

# Writing the csv lines
with open(csv_file, mode="w") as csv_f:
    csv_writer = csv.writer(
        csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    for line in csv_output:
        csv_writer.writerow(line)

# give some stats
durs = [float(row[1]) for row in csv_output[1:]]
print(f'{len(csv_output)-1} noise samples, mean duration {np.mean(durs):.2f} sec')
 
#TODO: possibly cut down ISAT noise, and have a separate ISAT noise adding step in the pipeline
# because MUSAN outnumbers ISAT by n files