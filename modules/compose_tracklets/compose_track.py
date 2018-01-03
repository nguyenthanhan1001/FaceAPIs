# settings
MIN_DURATION = 20
MIN_GAP = 5
NUM_PID = 69

import os
import sys

tracks = []

def split_line(line):
	hhmm, coor, pid = line.strip().split('\t')
	hh, mm = hhmm.split(':')
	hh = int(hh)
	mm = int(mm)

	coor = coor.split(' ')
	for i in range(4):
		coor[i] = int(coor[i])

	pid = int(pid)

	return hh, mm, coor, pid

def timespan(hhmm, h2, m2):
	t1 = hhmm[0] * 60 + hhmm[1]
	t2 = h2 * 60 + m2
	return t2 - t1

def duration(track):
	hh, mm = track['et']
	return timespan(track['st'], hh, mm)

def broaden_tracks(hh, mm, pid):
	if tracks[pid][-1].has_key('et'):
		if timespan(tracks[pid][-1]['et'], hh, mm) <= MIN_GAP:
			tracks[pid][-1]['et'] = (hh, mm)
		else:
			if duration(tracks[pid][-1]) > MIN_DURATION:
				tracks[pid].append({})
				tracks[pid][-1]['st'] = (hh, mm)
				tracks[pid][-1]['et'] = (hh, mm)
			else:
				tracks[pid][-1]['st'] = (hh, mm)
				tracks[pid][-1]['et'] = (hh, mm)
	else:
		tracks[pid][-1]['st'] = (hh, mm)
		tracks[pid][-1]['et'] = (hh, mm)

if __name__ == '__main__':
	filename = sys.argv[1]

	for it in range(NUM_PID):
		tracks.append([])
		tracks[it].append({})

	f = open(filename, "rt")
	for line in f:
		hh, mm, _, pid = split_line(line)
		broaden_tracks(hh, mm, pid)
		print '.',
	f.close()

	out_dir = os.path.dirname(filename) + '/tracklets_' + os.path.basename(filename).replace('.txt', '')
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
		print '\n%s created...'%(out_dir)
	else:
		print '\n%s existed...'%(out_dir)
	for pid in range(NUM_PID):
		g = open('%s/%d.txt'%(out_dir, pid), 'wt')
		for it in tracks[pid]:
			if not it.has_key('st'):
				continue
			st = it['st']
			et = it['et']
			g.write("%02d:%02d\t%02d:%02d\n"%(st[0], st[1], et[0], et[1]))
		g.close()
		print "%d.txt created..."%(pid)
