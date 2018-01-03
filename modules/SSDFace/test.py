import cv2
import FaceDetection as FD

fd = FD.FaceDetection()

# filename
f = open('../testset.txt', 'r')

for line in f:
	g = open('test.txt', 'a')

	img = cv2.imread(line[:-1])
	_, scores, bboxes = fd.dectectFace(img)
	g.write('%s\t%d\n'%(line[:-1], len(scores)))
	print('%s\t%d\n'%(line[:-1], len(scores)))
	for i in range(len(scores)):
		y1, x1, y2, x2 = bboxes[i]
		y = int(y1 * img.shape[0])
		x = int(x1 * img.shape[1])
		w = int((x2 - x1) * img.shape[1])
		h = int((y2 - y1) * img.shape[0])
		g.write('%.4f\t%d %d %d %d\n'%(scores[i], x, y, w, h))
		print('%.4f\t%d %d %d %d\n'%(scores[i], x, y, w, h))
		
	g.close()
f.close()