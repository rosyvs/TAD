import math
from tabnanny import check
from scipy.io import wavfile
import numpy as np
import glob
import os
from tqdm import tqdm
import contextlib
import re
from scipy.io import wavfile
import wave
import csv
import pandas as pd

# misc utils for data prep

def split_to_chunks(chunk_sec, total_sec, srate):
    """
    Returns list of chunks where each chunk is a list [start_sample, end_sample, duration_sample]
    """
    num_chunks = math.ceil(total_sec / chunk_sec)  
    chunk_list = []
    for i in range(num_chunks):
        chunk = [
            int(i * chunk_sec * srate) ,
            int(min(srate * (i * chunk_sec + chunk_sec), srate * total_sec))
        ]
        chunk.append(chunk[1] - chunk [0])
        chunk_list.append(chunk)
    # print(chunk_list)
    return chunk_list

def check_valid_wav(wav_file):
    samplerate, data = wavfile.read(wav_file)
    OK = True
    if data.dtype != np.dtype('int16'):
        print(f'Invalid wav file: dtype {data.dtype}  ({wav_file})')
        OK = False
    if any(np.isnan(data)):
        print(f'Invalid wav file : nan values  ({wav_file})')
        OK = False
    if min(data) < -32768 or max(data)> 32767:
        print(f'Invalid wav file: values exceed [-32768, 32767]  ({wav_file})')
    return OK

def collect_wav_metadata_from_fname(
    corpus_dir, 
    pattern, 
    csv_out, 
    srate=16000, 
    fixed_dur_sec=None, 
    keep_shorter=True, 
    chunk_sec=None, 
    check_wav=False):
    
    wav_lst = glob.glob(os.path.join(corpus_dir,'**/*.wav'), recursive=True)
    print(f'detected {len(wav_lst)} .wav files in "{corpus_dir}". Collecting metadata for csv...')

    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
    if check_wav:
        wav_error_files=[]

    for wav_file in tqdm(wav_lst):
        # Getting sentence and speaker ids
        try:
            m = re.search(fr'{pattern}', wav_file)
            mdict = m.groupdict()
            speaker = mdict.get('speaker','')
            uttID = mdict.get('uttID','')
            recordingID =  mdict.get('recordingID','')
            supID=mdict.get('supID','')
            subID=mdict.get('subID','')
            recordingID = ''.join(supID,recordingID,subID)
        except:
            print(f"Malformed path or pattern: {wav_file}")
            continue
        ID = '_'.join([recordingID, speaker, uttID])
        # print(ID)
        try:
            with contextlib.closing(wave.open(wav_file,'r')) as f:
                srate = f.getframerate()
                duration = f.getnframes()
                duration_sec = f.getnframes()/srate

            # check for invalid values
            if check_wav:
                samplerate, data = wavfile.read(wav_file)
                if not check_valid_wav(wav_file):
                    continue

        except EOFError:
            #print(f'invalid .wav file: {wav_file}')
            if check_wav:
                wav_error_files +=wav_file
            continue
            
        if srate != srate:
            raise ValueError(f'sampling rate is {srate}, but {srate} is required. Go back and reformat this corpus.')

        if fixed_dur_sec is not None:
            if duration_sec<fixed_dur_sec:
                if keep_shorter:
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
                else:
                    print(f'file too short ({duration_sec}) for fixed_dur_sec {fixed_dur_sec}, skipping')
                continue
            else:
                chunks = split_to_chunks(fixed_dur_sec, duration_sec, srate)
                csv_line = [[
                    f'{ID}_chunk{i:03}',
                    c[0],
                    c[1],
                    c[2],
                    speaker,
                    wav_file
                ] for i,c in enumerate(chunks) if keep_shorter or c[2]==int(fixed_dur_sec*srate)]
                csv_output.extend(csv_line)
        elif duration_sec>chunk_sec:
            #print(f'Long file: splitting into {math.ceil(duration_sec/chunk_sec)} segments of <={chunk_sec} seconds')
            chunks = split_to_chunks(chunk_sec, duration_sec, srate)
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

    # give some stats
    durs = [row[3] for row in csv_output]
    spkrs = [row[4] for row in csv_output]
    durs=durs[1:]
    spkrs = spkrs[1:]
    print(f'{len(csv_output)-1} utterances, mean duration {np.mean(durs)/srate:.2f} sec, {len(set(spkrs))} speakers')
    # Writing the csv lines
    print('...writing CSV')
    with open(csv_out, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    return wav_error_files if check_wav else None 

def collect_wav_metadata_from_csv(
    corpus_dir, 
    metadata_csv, 
    csv_out, 
    ID_key = 'ID',
    duration_key = "duration",
    duration_sec_key = "duration_sec",
    speaker_key = 'speaker',
    wav_file_key = 'filepath',
    srate=16000, 
    fixed_dur_sec=None, 
    keep_shorter=True, 
    chunk_sec=None, 
    check_wav=False):
    
    wav_lst = glob.glob(os.path.join(corpus_dir,'**/*.wav'), recursive=True)
    print(f'detected {len(wav_lst)} .wav files in "{corpus_dir}". Collecting metadata for csv...')

    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]
    if check_wav:
        wav_error_files=[]
    
    metadata = pd.read_csv(metadata_csv)
    for i,row in metadata.iterrows(): #TODO
        # Getting sentence and speaker ids
        try:
            duration = row.get(duration_key)
            duration_sec = row.get(duration_sec_key)
            ID = row.get(ID_key)
            speaker = row.get(speaker_key)
            wav_file = os.path.join(corpus_dir ,row.get(wav_file_key))
        except:
            print(f"Malformed path or pattern: {wav_file}")
            continue
        # print(ID)
            # check for invalid values
        if check_wav:
            if not check_valid_wav(wav_file):
                wav_error_files +=wav_file
                continue
        if fixed_dur_sec is not None:
            if duration_sec<fixed_dur_sec:
                if keep_shorter:
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
                else:
                    print(f'file too short ({duration_sec}) for fixed_dur_sec {fixed_dur_sec}, skipping')
                continue
            else:
                chunks = split_to_chunks(fixed_dur_sec, duration_sec, srate)
                csv_line = [[
                    f'{ID}_chunk{i:03}',
                    c[0],
                    c[1],
                    c[2],
                    speaker,
                    wav_file
                ] for i,c in enumerate(chunks) if keep_shorter or c[2]==int(fixed_dur_sec*srate)]
                csv_output.extend(csv_line)
        elif duration_sec>chunk_sec:
            #print(f'Long file: splitting into {math.ceil(duration_sec/chunk_sec)} segments of <={chunk_sec} seconds')
            chunks = split_to_chunks(chunk_sec, duration_sec, srate)
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

    # give some stats
    durs = [row[3] for row in csv_output]
    spkrs = [row[4] for row in csv_output]
    durs=durs[1:]
    spkrs = spkrs[1:]
    print(f'{len(csv_output)-1} utterances, mean duration {np.mean(durs)/srate:.2f} sec, {len(set(spkrs))} speakers')
    # Writing the csv lines
    print('...writing CSV')
    with open(csv_out, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    return wav_error_files if check_wav else None 