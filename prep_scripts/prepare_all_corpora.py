from distutils.command.check import check
import os
# from pydub import AudioSegment
import re
import numpy as np
from prep_utils import  collect_wav_metadata_from_csv,  collect_wav_metadata_from_fname
# from sphfile import SPHFile
from multiprocessing import Pool

# An easy way to run this over all files under the current directory recursively is find . -name '*.WAV' -exec sph2pipe -f wav {} {}.wav \;. The only drawback is that you end up with files ending with .WAV.wav
VERSTR = '_fixlen3'
CORPORA_PATH = '/mnt/shared/CORPORA/'
corpus_dir = '' 
srate = 16000
fixed_dur_sec = 3.0 # None or float, if float, utterances will be excluded if less than this, and truncated/chunked if more
keep_shorter = True # If False, segs shorter than fixed_dur_sec are skipped. Otherwise kept, will be 0-padded in model.
chunk_sec = 10.0 # max segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments
    # overriden by fixed_dur_sec if not None
CHECK_WAV = False

# make dir for manifest .csv files
manifest_dir = os.path.join(CORPORA_PATH,'data_manifests')
os.makedirs(manifest_dir, exist_ok=True)

dirs = {
    'cukids_train_part1': 'CuKidsSpeech/train/train-part1-cu',
    'cukids_train_part2': 'CuKidsSpeech/train/train-part2-cu-sentences',
    'cukids_train_part3': 'CuKidsSpeech/train/train-part3-cu-stories',
    'cukids_train_part4': 'CuKidsSpeech/train/train-part4-cu-summaries',
    'cukids_train_part5': 'CuKidsSpeech/train/train-part5-ogi-1-5',
    'cslu_scripted'     : 'cslu_kids/speech/scripted',
    'cslu_spontaneous'  : 'cslu_kids/speech/spontaneous',
    'myst_train'        : 'myst-v0.4.2/data/train', 
    'myst_development'  : 'myst-v0.4.2/data/development', 
    'myst_test'         : 'myst-v0.4.2/data/test'


}

# pattern can include named groups with the following names:
# speaker (required)
# recordingID OR supID and subID
# uttID

subdir_patterns = {
    'cukids_train_part1': r"CC-\d{2}-\d{2}-\d{5}\/(?P<recordingID>CC-\d{2}-\d{2})-(?P<speaker>\d{5})-(?P<uttID>\d{5}-\d*-\d*).wav",
    'cukids_train_part2': r"(?P<speaker>spk-\d{2}-\d{3})-sent\/(?P<uttID>\d*).wav",
    'cukids_train_part3': r"(?P<speaker>spk-\d{2}-\d{3})\w?\/(?P<uttID>[-\w\d]*-\d{3}).wav",
    'cukids_train_part4': r"(?P<speaker>spk-\d{2}-\d{3})\/(?P<uttID>[-\w\d]*-\d*).wav",
    'cukids_train_part5': r"ks\w{3}-\d{2}-\d{1}\/(?P<speaker>ks\w{3})(?P<uttID>\w{3}).wav",
    'cslu_scripted'     : r"(?P<supID>\d{2})\/(?P<subID>\d{1})\/(?P<speaker>[\w]*)\/\w{5}(?P<uttID>[\w]*).wav",
    'cslu_spontaneous'  : r"(?P<supID>\d{2})\/(?P<subID>\d{1})\/(?P<speaker>[\w]*)\/\w{5}(?P<uttID>[\w]*).wav",
    'myst_train'        : r"(?P<speaker>\d{6})\/(?P<recordingID>[\w\.\-\_]*)\/[\w\.-]*_(?P<uttID>[\w]*).wav", 
    'myst_development'  : r"(?P<speaker>\d{6})\/(?P<recordingID>[\w\.\-\_]*)\/[\w\.-]*_(?P<uttID>[\w]*).wav", 
    'myst_test'         : r"(?P<speaker>\d{6})\/(?P<recordingID>[\w\.\-\_]*)\/[\w\.-]*_(?P<uttID>[\w]*).wav"  
}

for name, path in dirs.items():
    collect_wav_metadata_from_fname(
        corpus_dir=os.path.join(CORPORA_PATH,path),
        pattern=subdir_patterns.get(name),
        csv_out = os.path.join(manifest_dir, f'{name}{VERSTR}.csv'),
        fixed_dur_sec=3.0)

# do ISAT by reading metadata csv
isat = {
'ISAT-SI_TRAIN': 'ISAT-SI/TRAIN/' ,
'ISAT-SI_TEST': 'ISAT-SI/TEST/' ,
'ISAT-SI_DEV':  'ISAT-SI/DEV/' 
}

for name,path in isat.items():
    collect_wav_metadata_from_csv(
    corpus_dir=os.path.join(CORPORA_PATH,'ISAT-SI'),
    metadata_csv = os.path.join(CORPORA_PATH,path,'METADATA.csv'), 
    csv_out=os.path.join(manifest_dir, f'{name}{VERSTR}.csv') ,
    fixed_dur_sec=3.0

    )

