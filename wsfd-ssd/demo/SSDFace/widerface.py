import FaceDetection
import cv2
import glob
import os, sys, time
import visualization
faceDect = FaceDetection.FaceDetection()

INPUT = '/home/nptai/WIDER_val/images'
OUTPUT = './output-widerface'

paths = glob.glob(os.path.join(INPUT,'*/*.jpg'))
paths.sort()

if not os.path.isdir(OUTPUT):
    os.mkdir(OUTPUT)


total = len(paths)
curr = 0
xxtime = time.time()
for path in paths:
    xtime = time.time()
    curr += 1

    img = cv2.imread(path)
    [w, h, _] = img.shape

    classes, scores, bboxes = faceDect.dectectFace(img)


    visualization.bboxes_draw_on_img(img, classes, scores, bboxes, visualization.colors, class_names=['none-face', 'face'])
    
    folder = path.split('/')[-2]
    filename = path.split('/')[-1].split('.')[0]
    
    dir = os.path.join(OUTPUT, folder)

    if not os.path.isdir(dir):
        os.mkdir(dir)

    
    outfile = os.path.join(OUTPUT, folder, '%s.txt' % filename)

    cv2.imwrite(outfile.replace('txt', 'jpg'), img)


    out = '%s\n%d' % (filename, len(scores))
    
    for i in range(len(scores)):
        [a, b, c, d] = bboxes[i]
        a = int(a*w)
        b = int(b*h)
        c = int(c*w)
        d = int(d*h)
        out = '%s\n%d %d %d %d %f' % (out, a, b, c-a, d-b, scores[i])

    open(outfile, 'w').write(out)

    sys.stdout.write('\r>>Processing image %d/%d: %f' % (curr, total, time.time() - xtime))
    sys.stdout.flush()

print 'Total time: %f' % (time.time() - xxtime)
