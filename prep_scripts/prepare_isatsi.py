import os
import csv
from tqdm import tqdm
from prep_utils import split_to_chunks
import numpy as np
import pandas as pd
import math

CORPORA_PATH = '/mnt/shared/CORPORA/'
CORPUS_DIR = 'ISAT-SI/' 
SRATE = 16000
CHUNK_SEC = 10 # segment duration in seconds (3.0 used in speechbrain recipe) 
    # None: untrimmed, variable-duration inputs / float: split into segments
splits = ['DEV','TEST','TRAIN']


os.makedirs(os.path.join(CORPORA_PATH,'data_manifests'), exist_ok=True)

for split in splits:
    csv_file = os.path.join(CORPORA_PATH,'data_manifests',f'ISAT-SI_{split}.csv')

    metadatafile = os.path.join(CORPORA_PATH, CORPUS_DIR, split, 'METADATA.csv')
    metadata = pd.read_csv(metadatafile)
    print(f'{len(metadata)} .wav files in split "{split}". Collecting metadata for csv...')

    csv_output = [["ID", "start", "end", "duration", "speaker", "filepath"]]

    for i,row in metadata.iterrows(): #TODO
        duration = row['duration']
        duration_sec = row['duration_sec']
        ID = row['ID']
        speaker = row['speaker']
        wav_file = os.path.join(CORPORA_PATH, CORPUS_DIR ,row['filepath'])
        if row['duration_sec']>CHUNK_SEC:
            print(f'Long file: splitting into {math.ceil(duration_sec/CHUNK_SEC)} segments of <={CHUNK_SEC} seconds')
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



# TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
# TODO: make verifiaction pairs file from TEST set

# # from voxceleb_prepare.py:

# def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
#     """
#     Creates the csv file for test data (useful for verification)

#     Arguments
#     ---------
#     data_folder : str
#         Path of the data folders
#     save_folder : str
#         The directory where to store the csv files.

#     Returns
#     -------
#     None
#     """

#     # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
#     # logger.debug(msg)

#     csv_output_head = [
#         ["ID", "duration", "wav", "start", "stop", "spk_id"]
#     ]  # noqa E231

#     for data_folder in data_folders:

#         test_lst_file = verification_pairs_file

#         enrol_ids, test_ids = [], []

#         # Get unique ids (enrol and test utterances)
#         for line in open(test_lst_file):
#             e_id = line.split(" ")[1].rstrip().split(".")[0].strip()
#             t_id = line.split(" ")[2].rstrip().split(".")[0].strip()
#             enrol_ids.append(e_id)
#             test_ids.append(t_id)

#         enrol_ids = list(np.unique(np.array(enrol_ids)))
#         test_ids = list(np.unique(np.array(test_ids)))

#         # Prepare enrol csv
#         logger.info("preparing enrol csv")
#         enrol_csv = []
#         for id in enrol_ids:
#             wav = data_folder + "/wav/" + id + ".wav"

#             # Reading the signal (to retrieve duration in seconds)
#             signal, fs = torchaudio.load(wav)
#             signal = signal.squeeze(0)
#             audio_duration = signal.shape[0] / SAMPLERATE
#             start_sample = 0
#             stop_sample = signal.shape[0]
#             [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

#             csv_line = [
#                 id,
#                 audio_duration,
#                 wav,
#                 start_sample,
#                 stop_sample,
#                 spk_id,
#             ]

#             enrol_csv.append(csv_line)

#         csv_output = csv_output_head + enrol_csv
#         csv_file = os.path.join(save_folder, ENROL_CSV)

#         # Writing the csv lines
#         with open(csv_file, mode="w") as csv_f:
#             csv_writer = csv.writer(
#                 csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
#             )
#             for line in csv_output:
#                 csv_writer.writerow(line)

#         # Prepare test csv
#         logger.info("preparing test csv")
#         test_csv = []
#         for id in test_ids:
#             wav = data_folder + "/wav/" + id + ".wav"

#             # Reading the signal (to retrieve duration in seconds)
#             signal, fs = torchaudio.load(wav)
#             signal = signal.squeeze(0)
#             audio_duration = signal.shape[0] / SAMPLERATE
#             start_sample = 0
#             stop_sample = signal.shape[0]
#             [spk_id, sess_id, utt_id] = wav.split("/")[-3:]

#             csv_line = [
#                 id,
#                 audio_duration,
#                 wav,
#                 start_sample,
#                 stop_sample,
#                 spk_id,
#             ]

#             test_csv.append(csv_line)

#         csv_output = csv_output_head + test_csv
#         csv_file = os.path.join(save_folder, TEST_CSV)

#         # Writing the csv lines
#         with open(csv_file, mode="w") as csv_f:
#             csv_writer = csv.writer(
#                 csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
#             )
#             for line in csv_output:
#                 csv_writer.writerow(line)
