import math
from scipy.io import wavfile
import numpy as np
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