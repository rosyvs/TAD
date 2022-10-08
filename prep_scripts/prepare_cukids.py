import glob
import os
import wave
import csv
from tqdm import tqdm
from pydub import AudioSegment
import contextlib
import re
import numpy as np
from prep_utils import split_to_chunks
from sphfile import SPHFile
from multiprocessing import Pool

# TODO: these are not actual wav files! They are NIST SPHERE which also uses the .wav ext...
# An easy way to run this over all files under the current directory recursively is find . -name '*.WAV' -exec sph2pipe -f wav {} {}.wav \;. The only drawback is that you end up with files ending with .WAV.wav

CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'CuKidsSpeech/train/' 
SRATE = 16000
CHUNK_SEC = 10 # segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments

# CUKIDS is futher split into subdirs of different kinds of audio, with different file/dir naming structure 
# represented by the regexp  in the dict below
subdir_patterns = {
   'train-part1-cu' : r"CC-\d{2}-\d{2}-\d{5}\/(?P<recordingID>CC-\d{2}-\d{2})-(?P<speaker>\d{5})-(?P<uttID>\d{5}-\d*-\d*).wav",
   'train-part2-cu-sentences': r"(?P<speaker>spk-\d{2}-\d{3})-sent\/(?P<uttID>\d*).wav",
   'train-part3-cu-stories': r"(?P<speaker>spk-\d{2}-\d{3})\w?\/(?P<uttID>[-\w\d]*-\d{3}).wav",
   'train-part4-cu-summaries': r"(?P<speaker>spk-\d{2}-\d{3})\/(?P<uttID>[-\w\d]*-\d*).wav",
   'train-part5-ogi-1-5': r"ks\w{3}-\d{2}-\d{1}\/(?P<speaker>ks\w{3})(?P<uttID>\w{3}).wav"
}


os.makedirs(os.path.join(CORPORA_PATH,'data_manifests'), exist_ok=True)

#######################
# TODO: sort out this file conversion ! Might have to use sox? here's an ancient help page 
# http://www.cs.columbia.edu/~ecooper/tts/raw.html


CONVERT_FILES = False # convert files from sph to wav, only needs to be done once
# note that this is particularly problematic because the original files are
# saved as '.wav' whereas sphere files should be saved as upeprvase ".WAV"
# so you can't tell whether htis is done just by listing filenames
# instead, run file <wavfilename> in linux terminal to see file type details

def sph2wav(filename):
    sph = SPHFile(filename)  
    sph.write_wav(filename)

if CONVERT_FILES:
    sph_files = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,'**/*.wav'), recursive=True)
    print(f'detected {len(sph_files)} .wav files. Converting...')

    convert_count =len(sph_files)
    with Pool(processes=min(convert_count, os.cpu_count())) as p:
        progress_bar = tqdm(total=convert_count)
        r = tqdm(p.imap(sph2wav, sph_files),total=convert_count)
        tuple(r)
        print('done')
########################
wav_error_files=[]
for split, pattern in subdir_patterns.items():
    csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'cukids_{split}.csv')

    wav_lst = glob.glob(os.path.join(CORPORA_PATH,CORPUS_DIR,split,'**/*.wav'), recursive=True)
    print(f'detected {len(wav_lst)} .wav files in split "{split}". Collecting metadata for csv...')

    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
    for wav_file in tqdm(wav_lst):
        # Getting sentence and speaker ids
        try:
            m = re.search(fr'{pattern}', wav_file)
            mdict = m.groupdict()
            speaker = mdict.get('speaker','')
            uttID = mdict.get('uttID','')
            recordingID =  mdict.get('recordingID','')
        except:
            print(f"Malformed path: {wav_file}")
            continue
        ID = '_'.join([split, recordingID,  speaker, uttID])

        try:
            with contextlib.closing(wave.open(wav_file,'r')) as f:
                srate = f.getframerate()
                duration = f.getnframes()
                duration_sec = f.getnframes()/srate
        except EOFError:
            #print(f'invalid .wav file: {wav_file}')
            wav_error_files +=wav_file
            continue
            
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

