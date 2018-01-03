import sys
sys.path.insert(0, '../SSDFace/')
import cv2
import os


import FaceDetection as FD
import VggRecogniser as VR

VIDEO_PATH = "/home/mmhci_hcmus/data/surv/surv.mp4"
OUT_DIR = "./"
THRES_DET = 0.8
THRES_REG = 0.8
THRES_SIZE = 0.1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
vout = cv2.VideoWriter('/home/mmhci_hcmus/data/surv/selab-3.avi', fourcc, 30, (1280, 720))

uids = ['Niem Bui', 'Tri Huynh', 'Dang Duong']

def run():
	fd = FD.FaceDetection()
	fr = VR.VggRecogniser()

	fout = createOutFile(VIDEO_PATH, OUT_DIR)
	if not os.path.exists(VIDEO_PATH):
		print 'video not exist...'
		return
	video_capture = cv2.VideoCapture(VIDEO_PATH)
	if not video_capture.isOpened():
		print 'video reading error...'
		return
	fps = int(video_capture.get(cv2.CAP_PROP_FPS))
	print 'fps: ', fps
	num_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
	print 'frame count: ', num_frame

	for count in range(num_frame):
		ret, frame = video_capture.read()
		#if not ret or count % fps != 0:
		if not ret:
			continue

		_, scores, bboxes = fd.dectectFace(frame)
		imgs = []
		coors = []
		for i in range(len(scores)):
			if scores[i] >= 0.8:
				y1, x1, y2, x2 = bboxes[i]
				y1 = int(y1 * frame.shape[0])
				y2 = int(y2 * frame.shape[0])
				x1 = int(x1 * frame.shape[1])
				x2 = int(x2 * frame.shape[1])
				w = x2 - x1
				h = y2 - y1
				#if w > frame.shape[1] * THRES_SIZE and h > frame.shape[0] * THRES_SIZE:
				if w > 124 and h > 124:
					face = frame[y1:y2, x1:x2, :]
					imgs.append(face)
					coors.append((x1, y1, x2-x1, y2-y1))
		if len(imgs) > 0:
			num, res = fr.recognise(imgs)
			ss = count / fps
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

					# write frame
					x, y, w, h = coor
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
					cv2.putText(frame, uids[uid],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
								0.5, (0,255,0), 1, cv2.LINE_AA)
		vout.write(frame)
		print '.'
	fout.close()



def createOutFile(video_path, out_dir):
	name = os.path.basename(video_path)
	out_filename = out_dir + name.replace('.mp4', '-meta') + '.txt'
	f = open(out_filename, 'a')
	return f


if __name__ == "__main__":
	run()
	vout.release()
