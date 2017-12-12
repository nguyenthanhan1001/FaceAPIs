import os, glob
import math
import random
import cv2

import numpy as np
import tensorflow as tf
import time
import visualization

import sys
sys.path.insert(0, '../')

slim = tf.contrib.slim


from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
tf.logging.set_verbosity(tf.logging.ERROR)

FACE_MODEL = '/var/www/SSDFace/model/model.ckpt-70000'
PERSON_MODEL= '/var/www/SSDFace/model/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

class _PersonDetection(object):
    def __init__(self):
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.InteractiveSession(config=config)
     
    def load(self, net_shape=(300, 300), data_format='NHWC', ckpt_filename=PERSON_MODEL):
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)

        # Define the SSD model.
        ssd_net = ssd_vgg_300.SSDNet()

        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=False)

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_filename)
        
        # SSD default anchor boxes.
        reuse = True
        self.ssd_anchors = ssd_net.anchors(net_shape)
        
    # Main image processing routine.
    def predict(self, img, select_threshold=0.05, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes 

class FaceDetection(object):
    def __init__(self):
        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.pdect = _PersonDetection()
        self.sess = tf.InteractiveSession(config=config)
        self.load()
        
    def load(self, net_shape=(300, 300), data_format='NHWC', ckpt_filename=FACE_MODEL):
        xtime = time.time()

        self.pdect.load()
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)

        # Define the SSD model.
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=True)

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_filename)
        
        # SSD default anchor boxes.
        reuse = True
        self.ssd_anchors = ssd_net.anchors(net_shape)
        
        print 'Load model in %3fs' % (time.time()-xtime)

    # Main image processing routine.
    def _predict(self, img, select_threshold=0.6, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.sess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes 

    def _area(self, b):
        b = b.reshape(2, 2)
        b = b[1] - b[0]
        return b[0]*b[1]

    def _check_overlap(self, b1, b2, thr):
        b3 = np.zeros(4)
        b4 = np.zeros(4)
        for i in range(4):
            if i < 2:
                b3[i] = min(b1[i], b2[i])
                b4[i] = max(b1[i], b2[i])
            else:
                b3[i] = max(b1[i], b2[i])
                b4[i] = min(b1[i], b2[i])
                
        s1 = self._area(b1)
        s2 = self._area(b2)
        s3 = self._area(b3)
        if b4[0] < b4[2] and b4[1] < b4[3]:
            s4 = self._area(b4)
        else:
            s4 = 0

        if s4/min(s1, s2) > thr or max(s1, s2)/s3 > thr:
            if s1 > s2:
                return b1
            else:
                return b2
        
        return []
                
    def _post_process_bbox(self, classes, scores, bboxes, r, thr):
        n = bboxes.shape[0]
        deleted = np.zeros(n)

        for i in range(n):
            if deleted[i] == 0:
                for j in range(n):
                    if i != j and deleted[j] == 0:
                        b = self._check_overlap(bboxes[i], bboxes[j], thr)
                        if len(b):
                            bboxes[i] = b
                            scores[i] = max(scores[i], scores[j])
                            deleted[j] = 1

        
        k = 0
        for i in range(n):
            if deleted[i] == 0:
                classes[k] = classes[i]
                scores[k] = scores[i]
                bboxes[k] = bboxes[i]

                for j in range(2):
                    t = (bboxes[k][2] - bboxes[k][0]) - (bboxes[k][3] - bboxes[k][1])*r
                    if t > 0:
                        bboxes[k][1] -= t/2/r
                        bboxes[k][3] += t/2/r
                    else:
                        bboxes[k][0] += t/2
                        bboxes[k][2] -= t/2

                    for t in range(4):
                        bboxes[k][t] = min(1, max(0, bboxes[k][t]))
                    
                k += 1
        
        return classes[:k], scores[:k], bboxes[:k]

    def dectectFace(self, img):
        [w, h, d] = img.shape    
        # img = mpimg.imread(path)
        rclasses, rscores, rbboxes =  self.pdect.predict(img)
        # visualization.plt_bboxes(img, rclasses, rscores, rbboxes, figsize=(25, 25))\
        # print len(rbboxes)
        # return rclasses, rscores, rbboxes
        bboxes = np.array([], dtype=np.int32)
        classes = np.array([], dtype=np.int32)
        scores = np.array([], dtype=np.int32)

        
        cnt = 0
        # print rbboxes.shape
        rclasses = np.concatenate((rclasses, np.array([15])))
        rbboxes = np.concatenate((rbboxes, np.array([[0, 0, 1, 1]])))
        for i in range(len(rbboxes)):
            if rclasses[i] == 15:
                [a, b, c, d] = rbboxes[i]
                x = 0.05
                tl = np.array([max(0, a-x), max(0, b-x)])
                
                a = max(0, int((a-x)*w))
                b = max(0, int((b-x)*h))
                c = min(w, int((c+x)*w))
                d = min(h, int((d+x)*h))

                sc = np.array([float(c-a)/w, float(d-b)/h])

                p = img[a:c, b:d]

                cs, ss, bs =  self._predict(p)
                bs = (bs.reshape(-1).reshape(-1, 2)*sc+tl).reshape(-1)
                classes = np.concatenate((classes, cs))
                scores = np.concatenate((scores, ss))
                bboxes = np.concatenate((bboxes, bs))
                cnt += 1

        bboxes = bboxes.reshape(-1, 4)
        classes, scores, bboxes = self._post_process_bbox(classes, scores, bboxes, float(h)/w, 0.6)
        # print cnt
        return classes, scores, bboxes


if __name__ == '__main__':
    fd = FaceDetection()
    # fd.load()
    img = cv2.imread('demo2.jpg')
    c, s, b = fd.dectectFace(img)
    visualization.bboxes_draw_on_img(img, c, s, b, visualization.colors, class_names=['none-face', 'face'])
    cv2.imwrite('demo2-out.jpg', img)
    


