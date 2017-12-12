from SSDFace.FaceDetection import FaceDetection 
from SSDFace import visualization
import cv2
import time
fd = FaceDetection()
img = cv2.imread('demo.jpg')
xtime = time.time()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
c, s, b = fd.dectectFace(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

visualization.bboxes_draw_on_img(img, c, s, b, visualization.colors, class_names=['none-face', 'face'])
print 'Process in %3fs' % (time.time()-xtime)

cv2.imwrite('demo-out.jpg', img)
import skimage.io as skio
skio.imsave('demoskio.jpg', img)