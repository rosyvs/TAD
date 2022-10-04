import glob
import os
import wave
import csv
from tqdm import tqdm
from multiprocessing import Pool
from pydub import AudioSegment
from prep_utils import split_to_chunks
import math
import numpy as np
import contextlib

# prepare speechbrain compatible csv annotation file 
# ID, start, end, duration, filepath, speaker (duration, start, end are in SAMPLES)

# ISAT-SI uses:
# index,segment,speaker,utterance,start_sec,end_sec,utterance_norm,duration_sec,duration,dataset,sessname,recordingID,ID,split,filepath


# MyST filestruct is {split}/{speaker}/{recordingID}/{recordingID}_{segment}.wav
# speakerID comes from the subfolder of split
# recordingID might contain '.'


CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'myst-v0.4.2/data/' 
SRATE = 16000
CHUNK_SEC = 30 # segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments
splits = ['train','development','test']

os.makedirs(os.path.join(CORPORA_PATH,'data_manifests'), exist_ok=True)

#######################
CONVERT_FILES = False # convert files from flac to wav, only needs to be done once

def flac2wav(filename):
    song = AudioSegment.from_file(filename).set_channels(1).set_sample_width(2).set_frame_rate(SRATE)
    song.export(filename.replace(".flac",".wav"), format = "wav")

if CONVERT_FILES:
    flac_files = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'*/*/*/*.flac'))
    print(f'detected {len(flac_files)} .flac files. Converting...')

    convert_count =len(flac_files)
    with Pool(processes=min(convert_count, os.cpu_count())) as p:
        progress_bar = tqdm(total=convert_count)
        r = tqdm(p.imap(flac2wav, flac_files),total=convert_count)
        tuple(r)
        print('done')
########################

for split in splits:
    csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'myst_{split}.csv')

    wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,f'{split}/*/*/*.wav'))
    print(f'detected {len(wav_lst)} .wav files in split "{split}". Collecting metadata for csv...')
    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
    # entry = []
    for wav_file in tqdm(wav_lst):
        # Getting sentence and speaker ids
        try:
            [speaker, recordingID, segmentID] = wav_file.split("/")[-3:]
        except ValueError:
            print(f"Malformed path: {wav_file}")
            continue
        ID = '_'.join([speaker, recordingID, segmentID.split(".")[0]])
        # get duration 
        with contextlib.closing(wave.open(wav_file,'r')) as f:
            srate = f.getframerate()
            duration = f.getnframes()
            duration_sec = f.getnframes()/srate
        
        if srate != SRATE:
            raise ValueError(f'sampling rate is {srate}, but {SRATE} is required. Go back and reformat this corpus.')
        if duration_sec>CHUNK_SEC:
            #print(f'Long file: splitting into {math.ceil(duration_sec/CHUNK_SEC)} segments of <={CHUNK_SEC} seconds')
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
    