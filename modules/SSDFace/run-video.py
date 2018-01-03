import os
import cv2

import FaceDetection as FD

OUT_DIR = '/home/mmhci_hcmus/output'
VIDEO_SET_FILE = '/home/mmhci_hcmus/data/QuocHoi/quochoi.txt'

if __name__ == '__main__':
	fv = open(VIDEO_SET_FILE)
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)

	fd = FD.FaceDetection()
	for path in fv:
		path = path.strip()
		video_capture = cv2.VideoCapture(path)
		frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
		fps = int(video_capture.get(cv2.CAP_PROP_FPS))
		name = os.path.basename(path).split('.')[0]
		outpath = os.path.join(OUT_DIR, name) 
		if not os.path.exists(outpath):
			os.mkdir(outpath)
		fout = open(outpath + '.txt', 'a')
		for ii in range(frame_count):
			ret, frame = video_capture.read()
			if not ret or ii % fps != 0:
				continue
			_, scores, bboxes = fd.dectectFace(frame)
			fout.write('%s\t%d\t%d\n'%(os.path.basename(path), ii, len(scores)))
			###
			print('%s\t%d\t%d'%(os.path.basename(path), ii, len(scores)))
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
				s_file = outpath + '/%s_%05d_%02d.jpg'%(name, ii, k)
				cv2.imwrite(s_file, s_img)
				
		fout.close()
	fv.close()
