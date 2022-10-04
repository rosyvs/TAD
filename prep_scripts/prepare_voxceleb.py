import glob
import os
import wave
import csv
from tqdm import tqdm
from prep_utils import split_to_chunks
import math
import numpy as np

# prepare speechbrain compatible csv annotation file 
# ID, start, end, duration, filepath, speaker (duration, start, end are in SAMPLES)

# ISAT-SI uses:
# index,segment,speaker,utterance,start_sec,end_sec,utterance_norm,duration_sec,duration,dataset,sessname,recordingID,ID,split,filepath

# ID, duration, wav, start, stop, spk_id are the columns used in the csvs for the pretrained Voxceleb 

# for consistency across our data, I will use 
# ID, start, end, duration, filepath, speaker (duration, start, end are in SAMPLES)

# VoxCeleb1 filestruct is wav/id00000/youtubeID/00000.wav
# speakerID comes from the subfolder of wav


CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'VoxCeleb/vox1_dev_wav/' 
SRATE = 16000
CHUNK_SEC = 30 # segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments


os.makedirs(os.path.join(CORPORA_PATH,'data_manifests'), exist_ok=True)
csv_file = os.path.join(CORPORA_PATH,'data_manifests','vox1_dev.csv')
# below is based on voxceleb_prepare.py in speechbrain recipes

# No split is required, will use all of dev to train the recogniser, tuning and testing will be done with ISAT-SI TEST
wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'*/*/*.wav'))
print(f'detected {len(wav_lst)} .wav files. Collecting metadata for csv...')
csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
for wav_file in tqdm(wav_lst, dynamic_ncols=True):
    # Getting sentence and speaker ids
    try:
        [speaker, recordingID, segmentID] = wav_file.split("/")[-3:]
    except ValueError:
        print(f"Malformed path: {wav_file}")
        continue
    ID = '_'.join([speaker, recordingID, segmentID.split(".")[0]])
    # get duration 
    with wave.open(wav_file,'rb') as f:
        srate = f.getframerate()
        duration = f.getnframes()
        duration_sec = f.getnframes()/srate

    if srate != SRATE:
        raise ValueError(f'sampling rate is {srate}, but {SRATE} is required. Go back and reformat this corpus.')
    if duration_sec>CHUNK_SEC:
        print(f'Long file: splitting into {math.ceil(duration_sec/CHUNK_SEC)} segments of <={CHUNK_SEC} seconds')
        chunks = split_to_chunks(CHUNK_SEC, duration_sec, SRATE)
        csv_line = [[
            ID,
            c[0],
            c[1],
            c[2],
            speaker,
            wav_file
        ] for c in chunks]
        csv_output.extend(csv_line)

    else:
        start=0
        end=duration
        # append to csv rows
        csv_line = [
            ID,
            start,
            end,
            duration,
            speaker,
            wav_file
        ]
        csv_output.append(csv_line)

# Writing the csv lines
with open(csv_file, mode="w") as csv_f:
    csv_writer = csv.writer(
        csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    for line in csv_output:
        csv_writer.writerow(line)

# give some stats
durs = [row[3] for row in csv_output]
spkrs = [row[4] for row in csv_output]
durs=durs[1:]
spkrs = spkrs[1:]
print(f'{len(csv_output)-1} utterances, mean duration {np.mean(durs)/SRATE:.2f} sec, {len(set(spkrs))} speakers')