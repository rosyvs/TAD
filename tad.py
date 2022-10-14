import torchaudio
from torch import tensor, cat, squeeze
import os
import re
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
import csv

# arguments and defaults
import argparse
parser = argparse.ArgumentParser(description='segment session files')
parser.add_argument('wav', help='path to wav file to process')
parser.add_argument('--targets', default='all')
parser.add_argument('--enrollments', default='../data/ISAT-SI/enrollments')
parser.add_argument('--frame_rate', default='16000',help='frames/sec')
parser.add_argument('--win', default='16000',help='window len frames')
parser.add_argument('--shift', default='4000',help='shift len frames')
parser.add_argument('--sbdir', default='../speechbrain/',help='sb dir')
parser.add_argument('--model', default='ecapa',help='ecapa or xvect')
parser.add_argument('--crit', default='0.1',help='threshold')
parser.add_argument('--minseg', default='8000',help='num frames to start seg')
parser.add_argument('--minsil', default='8000',help='num frames to start sil')
parser.add_argument('--outdir', default='segments_clip',help='output dir')
args = parser.parse_args()

#####
speechbrain_dir = args.sbdir
src_ecapa = "speechbrain/spkrec-ecapa-voxceleb"
models_ecapa = f"{speechbrain_dir}/pretrained_models/spkrec-ecapa-voxceleb"
src_xvect = "speechbrain/spkrec-xvect-voxceleb"
src_xvectv = "speechbrain/spkrec-xvect-voxcelebTEST"
models_xvect = f"{speechbrain_dir}/pretrained_models/spkrec-xvect-voxceleb"
#####

FRAME_DUR = float(1.0)/float(args.frame_rate)
WIN_SIZE = int(args.win)
SHIFT_LEN = int(args.shift)
CRIT = float(args.crit)
model_type = args.model
minseg = int(args.minseg)
minsil = int(args.minsil)
clip = os.path.splitext(os.path.basename(args.wav))[0]

# load models
if model_type == 'ecapa':
    encoder = EncoderClassifier.from_hparams(source=src_ecapa, 
              savedir=models_ecapa)
    verifier = SpeakerRecognition.from_hparams(source=src_ecapa, 
              savedir=models_ecapa)
if model_type == 'xvect':
    encoder = EncoderClassifier.from_hparams(source=src_xvect, 
        savedir=models_xvect)
    verifier = SpeakerRecognition.from_hparams(source=src_xvectv, 
        savedir=models_xvect)


# get list of target files
if args.targets == 'all':
    targs = []
    # get all student targets from transcript
    transfile=re.sub('WAVs','utt_labels',args.wav)
    transfile = re.sub('.wav','.csv',transfile)
    with open(transfile, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('Student'):
                name = re.sub('Student ','student-',row[0])
            elif row[0].startswith('student'):
                if row[0].startswith('student-other'): continue
                name = row[0]
            else: continue
            if not name in targs:
                targs.append(name)
else:
    targs = args.targets.split(',')
print(targs)

# get enrollments
enrollments = []
for name in targs:
    file = os.path.join(args.enrollments,'enrollment_%s.wav' % name)
    if not os.path.exists(file):
        print('target file not found:', file)
        exit(-1)
    enrollments.append(file)
print('enrollments')
print(enrollments)

# precompute embeddings for targets
target_embeddings = {}
for t in enrollments:
    signal, fs = torchaudio.load(t)
    if not fs == 16000:
        # fs must be 16000 
        to16k = torchaudio.transforms.Resample(fs, 16000)
        signal = to16k(signal)
    target_embeddings[t] = encoder.encode_batch(signal)

# get input file info
wavfile = args.wav
metadata = torchaudio.info(wavfile)
total_frames = metadata.num_frames
file_secs = float(total_frames)/16000.0
print(wavfile,'file len:',file_secs, ' sec')

# read input wav file
#signal, fs = torchaudio.load(wavfile)
#if not fs == 16000:
#    # fs must be 16000
#    to16k = torchaudio.transforms.Resample(fs, 16000)
#    signal = to16k(signal)

last_win_st = total_frames - WIN_SIZE  # start frame for last window

state = 'o'   # indicates whether inside or outside segment
seg_sf = 0 # start frame of first sample in segment
seg_ef = 0 # start frame of last sample in segment
# use sliding window of WIN_SIZE frames to generate samples
# compare each sample to all targets
# generate start and end times for target segments
segments = []
for win_st in range(0,last_win_st,SHIFT_LEN):
    # read next sample of WIN_SIZE frames
    win,fs= torchaudio.load(wavfile, frame_offset=win_st, num_frames=WIN_SIZE)
    xv_samp = encoder.encode_batch(win)

    # compare sample to each target
    scores = []
    for t in target_embeddings:
        score = verifier.similarity(target_embeddings[t], xv_samp)
        scores.append(score.item())

    # set sample score to 1 if any target score > CRIT
    maxscore = scores[0]
    frame_score = 0
    for s in scores:
        if s > CRIT: frame_score = 1
        if s > maxscore: maxscore = s
    #tim = '%f.1' % (FRAME_DUR * win_st)
    #print(tim, maxscore)

    # if inside segment
    if state == 'i':
        # if sample is target continue inside seg
        if frame_score:
            seg_ef = win_st
            continue

        # sample is not target
        # if seg too short, discard
        if (seg_ef - seg_sf) < minseg:
            state = 'o'
            continue

        # if sil too short, don't terminate seg
        if ((win_st - seg_ef) + WIN_SIZE) < minsil: continue

        # terminate and save segment
        # determine seg boundaries in secs
        # start mid first seg, end mid last seg
        seg_stsec = (seg_sf + (WIN_SIZE/2)) * FRAME_DUR
        seg_edsec = (seg_ef + (WIN_SIZE/2)) * FRAME_DUR
        seg = []
        seg.append(seg_stsec)
        seg.append(seg_edsec)
        # find best score speaker
        name = ''
        best_score = -100.0
        for sn, s in enumerate(scores):
            if s > best_score:
                name = targs[sn]
                best_score = s
        seg.append(name)
        for sn, s in enumerate(scores):
            seg.append('%s:%.3f' % (targs[sn],s))
        segments.append(seg)
        state = 'o'

    # if outside segment
    else:
        # if target segment start new seg
        if frame_score:
            seg_sf = win_st
            seg_ef = win_st
            state = 'i'
        # else continue outside

print(args.wav)
print('len(segments)', len(segments))
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
outf = os.path.join(args.outdir, '%s.csv' % clip)
with open(outf, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(segments)
