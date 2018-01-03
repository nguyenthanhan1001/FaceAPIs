import os
import cv2

import FaceDetection as FD

OUT_DIR = '/home/mmhci_hcmus/cropped-avatars'
IMAGES_FILE = '/home/mmhci_hcmus/avatars.txt'

if __name__ == '__main__':
	fi = open(IMAGES_FILE, 'r')
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)
	
	fd = FD.FaceDetection()
	fout = open(OUT_DIR + '.txt', 'w')
	for path in fi:
		path = path.strip()
		#print path
		name = os.path.basename(path).split('.')[0]
		_fd_ =  ''
		outpath = os.path.join(OUT_DIR, _fd_)
		#if os.path.exists(outpath):
		#	continue
		if not os.path.exists(outpath):
			os.mkdir(outpath)
		
		frame = cv2.imread(path)
		try:
			assert(len(frame.shape) == 3)
		except:
			fout.write('%s\t%d\n'%(os.path.basename(path), -1))
			print ('%s\t%d'%(os.path.basename(path), -1))
			continue
		_, scores, bboxes = fd.dectectFace(frame)
		fout.write('%s\t%d\n'%(os.path.basename(path), len(scores)))
		###
		print('%s\t%d'%(os.path.basename(path), len(scores)))
		for k in range(len(scores)):
			y1, x1, y2, x2 = bboxes[k]
			y = int(y1 * frame.shape[0])
			x = int(x1 * frame.shape[1])
			w = int((x2 - x1) * frame.shape[1])
			h = int((y2 - y1) * frame.shape[0])
			fout.write('%.4f\t%d %d %d %d\n'%(scores[k], x, y, w, h))
			###
			print('%.4f\t%d %d %d %d'%(scores[k], x, y, w, h))
			s_img = frame[y:y+h, x:x+w, :]
			s_file = outpath + '/%s_%02d.jpg'%(name, k)
			cv2.imwrite(s_file, s_img)
				
	fout.close()
	fi.close()
