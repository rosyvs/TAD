import glob
import os
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
SEG_DUR = 3.0 # segment duration (as per speechbrain recipe) TODO: can't training data be arbitraery duration?

# below is based on voxceleb_prepare.py in speechbrain recipes


# No split is required, will use all of dev to train the recogniser, tuning and testing will be done with ISAT-SI TEST
wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'*/*/*.wav'))
print(f'detected {len(wav_lst)} .wav files. Inferring metadata from path...')
for wav_file in tqdm(wav_lst, dynamic_ncols=True):
    # Getting sentence and speaker ids
    try:
        [speaker, recordingID, ID] = wav_file.split("/")[-3:]
        print(f'speaker: {speaker} || recordingID: {recordingID} || ID: {ID}')
    except ValueError:
        print(f"Malformed path: {wav_file}")
        continue
    # TODO: append to csv rows
    # TODO: decide if we need duration
    # TODO: check samplign rate of a few files? 
