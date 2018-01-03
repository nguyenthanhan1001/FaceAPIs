import sys
import cv2
import os

import VggRecogniser as VR

IN_DIR = '/home/mmhci_hcmus/output/detected'
IN_META = ['NA170605AM1', 'NA170605AM2', 'NA170614AM1', 'NA170614AM2', 'NA170615AM2']

OUT_DIR = "/home/mmhci_hcmus/output/recognised/"

THRES_DET = 0.8
THRES_REG = 0.8
THRES_SIZE = 124

def run(fr, images_dir, meta_path, video_name):
	fout = createOutFile(meta_path, OUT_DIR)
	if not os.path.exists(meta_path):
		print '%s not exist...'%(meta_path)
		return
	if not os.path.exists(images_dir):
		print '%s not exist...'%(images_dir)
		return

	fin = open(meta_path)
	count = -1
	while True:
		line = fin.readline()
		if len(line) < 1:
			break

		count += 1
		_, frame_id, num = line.strip().split('\t')
		frame_id = int(frame_id)
		num = int(num)

		imgs = []
		coors = []
		for i in range(num):
			score, xywh = fin.readline().strip().split('\t')
			x, y, w, h = xywh.split(' ')
			score = float(score)
			x = int(x)
			y = int(y)
			w = int(w)
			h = int(h)

			if w > THRES_SIZE and h >  THRES_SIZE and score > THRES_DET:
				pth = images_dir + '/' + video_name + '_%05d_%02d.jpg'%(frame_id, i)
				if not os.path.exists(pth):
					print pth, 'not exists...'
					raw_input()
				face = cv2.imread(pth)
				imgs.append(face)
				coors.append((x, y, w, h))
		if len(imgs) > 0:
			num, res = fr.recognise(imgs)
			ss = count
			mm = ss / 60
			ss = ss % 60
			for k in range(num):
				uid, scr = res[k]
				if scr >= THRES_REG:
					coor = coors[k]
					fout.write('%02d:%02d\t%d %d %d %d\t%s\n'%(
						mm, ss, 
						coor[0], coor[1], coor[2], coor[3],
						uid))
					print('%02d:%02d\t%d %d %d %d\t%s'%(
						mm, ss, 
						coor[0], coor[1], coor[2], coor[3],
						uid))
		print '.'

	fin.close()
	fout.close()

def createOutFile(meta_path, out_dir):
	name = os.path.basename(meta_path)
	out_filename = out_dir + name.replace('.txt', '_recognised.txt')
	f = open(out_filename, 'a')
	return f

if __name__ == "__main__":
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	fr = VR.VggRecogniser()
	for it in IN_META:
		imgs_dir = IN_DIR + '/' + it	
		run(fr, imgs_dir, imgs_dir + '.txt', it)