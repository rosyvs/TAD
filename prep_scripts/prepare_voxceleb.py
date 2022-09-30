import glob
import os
import wave
import csv
from tqdm import tqdm
# prepare speechbrain compatible csv annotation file 
# ISAT-SI uses:
# index,segment,speaker,utterance,start_sec,end_sec,utterance_norm,duration_sec,duration,dataset,sessname,recordingID,ID,split,filepath


# ID, duration, wav, start, stop, spk_id are the columns used in thecsvs for the pretrained Voxceleb 
# for consistency across our data, I will rename these to 
# ID, duration_sec, filepath, _, _, speaker (start and stop are unused as individual utts are already extracted)

# VoxCeleb1 filestruct is wav/id00000/youtubeID/00000.wav
# speakerID comes from the subfolder of wav


CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'VoxCeleb/vox1_dev_wav/' 
SRATE = 16000
SEG_DUR = None # segment duration in seconds (3.0 used in speechbrain recipe) None = untrimmed,variable-duration inputs
csv_file = './data_manifests/vox1_dev.csv'
os.makedirs('./data_manifests/', exist_ok=True)
# below is based on voxceleb_prepare.py in speechbrain recipes

# No split is required, will use all of dev to train the recogniser, tuning and testing will be done with ISAT-SI TEST
wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'*/*/*.wav'))
print(f'detected {len(wav_lst)} .wav files. Collecting metadata for csv...')
csv_output = [["ID", "duration", "speaker", "filepath"]]
entry = []
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
        if srate != SRATE:
            raise ValueError(f'sampling rate is {srate}, but {SRATE} is required. Go back and reformat this corpus.')
        duration = f.getnframes()
        # duration_sec = f.getnframes()/srate

    # start_sec=0
    # end_sec=duration_sec
    # append to csv rows
    csv_line = [
        ID,
        duration,
        speaker,
        wav_file
    ]
    entry.append(csv_line)
csv_output = csv_output+entry

# Writing the csv lines
with open(csv_file, mode="w") as csv_f:
    csv_writer = csv.writer(
        csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    for line in csv_output:
        csv_writer.writerow(line)

