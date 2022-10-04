import math

# misc utils for data prep

def split_to_chunks(chunk_sec, total_sec, srate):
    """
    Returns list of chunks where each chunk is a list [start_sample, end_sample, duration]
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