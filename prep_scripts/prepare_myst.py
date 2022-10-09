import glob
import os
import wave
import csv
from tqdm import tqdm
from multiprocessing import Pool
from pydub import AudioSegment
from prep_utils import check_valid_wav, split_to_chunks
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

VERSTR = '_fixlen1'

CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'myst-v0.4.2/data/' 
SRATE = 16000
FIXED_DUR_SEC = 1.0 # None or float, if float, utterances will be excluded if less than this, and truncated/chunked if more
CHUNK_SEC = 10.0 # max segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments
    # overriden by FIXED_DUR_SEC if not None
CHECK_WAV_VALID = False

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
    csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'myst_{split}{VERSTR}.csv')

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
        ID = '_'.join([speaker, recordingID, segmentID.replace(".wav","")])
        # get duration 
        with contextlib.closing(wave.open(wav_file,'r')) as f:
            srate = f.getframerate()
            duration = f.getnframes()
            duration_sec = f.getnframes()/srate
        if CHECK_WAV_VALID:
            if not check_valid_wav(wav_file):
                continue
        if srate != SRATE:
            raise ValueError(f'sampling rate is {srate}, but {SRATE} is required. Go back and reformat this corpus.')

        if FIXED_DUR_SEC:
            if duration_sec<FIXED_DUR_SEC:
                #print(f'file too short ({duration_sec}) for FIXED_DUR_SEC {FIXED_DUR_SEC}, skipping')
                continue
            else:
                chunks = split_to_chunks(FIXED_DUR_SEC, duration_sec, SRATE)
                csv_line = [[
                    f'{ID}_chunk{i:03}',
                    c[0],
                    c[1],
                    c[2],
                    speaker,
                    wav_file
                ] for i,c in enumerate(chunks) if c[2]==int(FIXED_DUR_SEC*SRATE)]
                csv_output.extend(csv_line)
        elif duration_sec>CHUNK_SEC:
            #print(f'Long file: splitting into {math.ceil(duration_sec/CHUNK_SEC)} segments of <={CHUNK_SEC} seconds')
            chunks = split_to_chunks(CHUNK_SEC, duration_sec, SRATE)
            csv_line = [[
                f'{ID}_chunk{i:03}',
                c[0],
                c[1],
                c[2],
                speaker,
                wav_file
            ] for i,c in enumerate(chunks)]
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
    