# align.py <segment.csv>
import os
import re
import csv
import wave

import argparse
parser = argparse.ArgumentParser(description='compare segments')
parser.add_argument('--seg', help='segments csv')
parser.add_argument('--sessdir', help='dir with session files')
parser.add_argument('--targets', default='all', help='comma separated set of target names')
parser.add_argument('--btol', default='0.0')
parser.add_argument('--ignoreoverlap', default='n')
args = parser.parse_args()

def score():
    global segments
    cls = ''
    st_sec = sg_st
    ed_sec = sg_ed
    dur = ed_sec - st_sec
    if dur <= 0.0: return

    if ref_state == 'i':
        if hyp_state == 'i':
            cls = 'CA'
            #name = ref_name
        else:
            cls = 'FR'
            #name = ref_name
    else:
        if hyp_state == 'i':
            cls = 'FA'
            #name = hyp_name
        else:
            cls = 'CR'
            #name = '-'

    # enforce boundary tolerance
    if (cls == 'FR'):
        if dur <= btol:
            cls = 'CR'
    if (cls == 'FA'):
        if dur <= btol:
            cls = 'CA'

    smin,ssec = divmod(sg_st, 60.0)
    emin,esec = divmod(sg_ed, 60.0)
    segments.append(('%i:%.1f' % (int(smin),ssec),\
                    '%i:%.1f' % (int(emin),esec), cls))
#    segments.append(('%i:%.1f' % (int(smin),ssec),\
#                    '%i:%.1f' % (int(emin),esec),\
#                    name, cls))

clip = os.path.splitext(os.path.basename(args.seg))[0]
print(clip)
transfile = os.path.join(args.sessdir, 'utt_labels/%s.csv' % clip)
# set boundary tolerance
btol = float(args.btol)

# get list of target files
target_names = []
if args.targets == 'all':
    # get all student targets from transcript
    with open(transfile, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('speaker'): continue
            if row[0].startswith('Student'):
                name = re.sub('Student ','student-',row[0])
            elif row[0].startswith('student'):
                if row[0].startswith('student-other'): continue
                name = row[0]
            else: continue
            if not name in target_names:
                target_names.append(name)
else:
    target_names = args.targets.split(',')


# read transcript segments
last_end = 0.0
ref_segs = []
name=''
with open(transfile, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].startswith('speaker'): continue
        # start time
        stsec=float(row[2])
        # end time
        edsec=float(row[3])

        # get speaker name
        name = ''
        if row[0].startswith('Student'):
            name = re.sub('Student ','student-',row[0])
        elif row[0].startswith('student'):
            if not row[0].startswith('student-other'): name = row[0]

        # get  text
        txt = row[1]
        # remove text inside brackets and parens
        txt = re.sub("[\(\[].*?[\)\]]", "", txt)
        txt = txt.strip()

        # ignore segment if no words or wrong speaker
        if (not txt) or (not name): continue
        if not name in target_names: continue

        ref_segs.append((stsec,edsec, name,txt))
        if edsec > last_end: last_end = edsec
        
# combine overlapping segments
new_segs = []
sg_st = 0.0
sg_ed = 0.0
sg_txt = ''
sg_spk = ''
for row in ref_segs:
    stsec = row[0]
    edsec = row[1]
    spk = row[2]
    txt = row[3]

    # overlap, combine with previous
    if stsec < sg_ed:
        if edsec > sg_ed: sg_ed = edsec
        sg_txt +=  ' ' + txt
    else:
        # output prev seg
        if sg_ed > 0.0:
            new_segs.append( (sg_st,sg_ed,sg_spk,sg_txt) )
        sg_st = stsec
        sg_ed = edsec
        sg_txt = txt
        sg_spk = spk
if sg_ed > 0.0: new_segs.append( (sg_st,sg_ed,sg_spk,sg_txt) )
print('len(new_segs)',len(new_segs))

# read hyp segments
hypf = args.seg
hyp_segs = []
with open(hypf, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        start = float(row[0])
        end = float(row[1])
        hyp_segs.append((start,end))
        #name = row[2]
        #hyp_segs.append((start,end,name))
        if end > last_end: last_end = end
if len(hyp_segs) == 0:
    print(args.seg,'no segments')
    exit()

# align segments to compute overlaps
# find time of first segment for ref and hyp
segments = []
sg_st = 0.0
ref_state = 'o'
ri = 0

nr = new_segs[ri][0]
ref_name = new_segs[ri][2]
nrs = 'i'

hyp_state = 'o'
hi = 0

nh = hyp_segs[hi][0]
#hyp_name = hyp_segs[hi][2]
nhs = 'i'

while sg_st < last_end:
    # find time of next state change from current pos
    if nh <= nr:
        sg_ed = nh
        score()
        hyp_state = nhs
        # find next hyp state change
        if hyp_state == 'i':
            nh = hyp_segs[hi][1]
            nhs = 'o'
        else:
            if hi < (len(hyp_segs)-1):
                hi += 1
                nh = hyp_segs[hi][0]
                #hyp_name = hyp_segs[hi][2]
                nhs = 'i'
            else:
                nh = last_end
                nhs = 'o'
    else:
        sg_ed = nr
        score()
        ref_state = nrs
        if ref_state == 'i':
            nr = new_segs[ri][1]
            nrs = 'o'
        else:
            if ri < (len(new_segs)-1):
                ri += 1
                nr = new_segs[ri][0]
                #ref_name = new_segs[ri][2]
                nrs = 'i'
            else:
                nr = last_end
                nrs = 'o'
    sg_st = sg_ed

# determine file length
wavf = os.path.join(args.sessdir, 'WAVs/%s.wav' % clip)
with wave.open(wavf) as mywav:
    duration_seconds = mywav.getnframes() / mywav.getframerate()
# add end silence to correct reject
end_sil = duration_seconds - last_end
if end_sil > 0.0:
    smin,ssec = divmod(last_end, 60.0)
    emin,esec = divmod(duration_seconds, 60.0)
    segments.append(('%i:%.1f' % (int(smin),ssec),\
                    '%i:%.1f' % (int(emin),esec),\
                    '-', 'CR'))

# write out segments
outf = re.sub('segments','alignments',args.seg)
aligndir=os.path.dirname(outf)
if not os.path.exists(aligndir):
    os.makedirs(aligndir)
with open(outf, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(segments)
