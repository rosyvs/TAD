import glob
import os
import wave
import csv
from tqdm import tqdm
from pydub import AudioSegment
from prep_utils import split_to_chunks
import numpy as np
import contextlib
# /mnt/shared/CORPORA/cslu_kids/speech/scripted/00/0/ks001/ks001{3 chars}.wav - multiple wav files
# /mnt/shared/CORPORA/cslu_kids/speech/spontaneous/00/0/ks001/ - single wav file


# readmes are not helpful regarding extraction of speakerID from this
# I would assume in the above example ks001 is the id. could count unique to verify, should be 1100
#

# ~~~~~~~About this corpus~~~~~~~
# spontaneous and prompted speech from 1100 children
# 1017 files containing approximately 8-10 minutes of speech per speaker.
# Scripted speech:  60 items from a total list of 319 phonetically-balanced but simple words, sentences or digit strings. 
# Each utterance of spontaneous speech begins with a recitation of the alphabet and contains a monologue of about one minute in duration.

CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'cslu_kids/speech/' 
SRATE = 16000
CHUNK_SEC = 30 # segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments
splits = ['scripted','spontaneous']

os.makedirs(os.path.join(CORPORA_PATH,'data_manifests'), exist_ok=True)

for split in splits:
    csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'cslu_{split}.csv')

    wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,f'{split}/*/*/*/*.wav'))
    print(f'detected {len(wav_lst)} .wav files in "{split}". Collecting metadata for csv...')
    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
    for wav_file in tqdm(wav_lst):
        # Getting sentence and speaker ids
        try:
            [supID, subID, speaker, segmentID] = wav_file.split("/")[-4:]
        except ValueError:
            print(f"Malformed path: {wav_file}")
            continue
        ID = '_'.join([subID, supID, segmentID.split(".")[0]])

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

