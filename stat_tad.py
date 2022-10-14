import os
import re
import csv
import pathlib

import argparse
parser = argparse.ArgumentParser(description='compare segments')
parser.add_argument('--aligned', help='aligned segments')
parser.add_argument('--exclude', default='')
args = parser.parse_args()


cls2idx = {'CA':0,'CR':1,'FA':2,'FR':3}

if args.exclude:
    exclude=args.exclude.split(',')

path=pathlib.Path(args.aligned)
outdir = path.parent

clip_stats = []
clip_stats.append( ('Clip','CR%','FR%','F','P','R','CA','CR','FA','FR'))

# for each clip in dir
stats = {}
for file in os.listdir(args.aligned):
    fname= os.path.join(args.aligned,file)
    with open(fname, newline="") as f:
        reader = csv.reader(f)
        for row in reader:

            if row[0].startswith('#'): continue
            # start and end time, convert from m:s to s
            m,s = row[0].split(':')
            stsec = float(m)*60 + float(s)
            m,s = row[1].split(':')
            edsec = float(m)*60 + float(s)

            if len(row) == 3:
                cls = row[2]
            else:
                cls = row[3]
            idx = cls2idx[cls]
            name = re.sub('.csv','',file)
            if not name in stats:
                stats[name] = [0.0,0.0,0.0,0.0]
            stats[name][idx] = stats[name][idx] + (edsec - stsec)

av_cr_rate = 0.0
av_fr_rate = 0.0
av_P = 0.0
av_R = 0.0
av_F = 0.0
count = 0
for clip in stats:
    ca = stats[clip][0]
    cr = stats[clip][1]
    fa = stats[clip][2]
    fr = stats[clip][3]
    noise_sec = cr + fa
    signal_sec = ca + fr
    cr_rate = 0.0
    if noise_sec > 0.0:
        cr_rate = cr/noise_sec
    fr_rate = 0.0
    if signal_sec > 0.0:
        fr_rate = fr/signal_sec
    if ca == 0:
        P = 0.0
        R = 0.0
        F = 0.0
    else:
        P = ca/(ca+fa)
        R = ca/(ca+fr)
        F = 2.0 * (P * R)/(P + R)
    clip_stats.append((clip,'%0.3f' % cr_rate,'%0.3f' % fr_rate,'%0.3f' % F,'%0.3f' % P,'%0.3f' % R,'%0.3f' % ca,'%0.3f' % cr,'%0.3f' % fa,'%0.3f' % fr))
    #print('%20s\t%5.1f\t%5.1f\t%5.1f\t%5.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f' % \
       #(clip,ca,cr,fa,fr,cr_rate,fr_rate,P,R,F))
    
    av_cr_rate += cr_rate
    av_fr_rate += fr_rate
    av_P += P
    av_R += R
    av_F += F
    count += 1

# write csv file
outstats = os.path.join(outdir,'stats.csv')
with open(outstats, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(clip_stats)

if count:
    av_cr_rate = av_cr_rate / count
    av_fr_rate = av_fr_rate / count
    av_P = av_P / count
    av_R = av_R / count
    av_F = av_F / count
else:
    av_cr_rate = 0.0
    av_fr_rate = 0.0
    av_P = 0.0
    av_R = 0.0
    av_F = 0.0

outsum = os.path.join(outdir,'summary')
with open(outsum,'w') as outfile:
    outfile.write('Average: CRr %0.3f  FRr  %0.3f  P %0.3f  R %0.3f  F %0.3f' % (av_cr_rate,av_fr_rate,av_P,av_R,av_F))
