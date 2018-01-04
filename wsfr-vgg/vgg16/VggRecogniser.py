import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python import learn
import vgg16
import utils
import cv2
import config

class VggRecogniser():
    def __init__(self, model_vgg_path = config.MODEL_VGG_PATH,\
                       model_dnn_path = config.MODEL_DNN_PATH, \
                       uid_path = config.UID_PATH):
        lr = 1e-4
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

        self.classifier = learn.DNNClassifier(
            model_dir=model_dnn_path,
            feature_columns=feature_columns,
            hidden_units=[4096, 2048],
            n_classes=530,
            activation_fn=tf.nn.relu,
            optimizer=tf.train.AdamOptimizer(learning_rate=lr)
        )

        self.sess = tf.Session()
        self.vgg = vgg16.Vgg16(model_vgg_path)
        self.images = tf.placeholder("float", [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            self.vgg.build(self.images)

        self.uid = []
        with open(uid_path, 'rt') as f:
            for it in f:
                self.uid.append(it.replace('\n', ''))

    def getname(self, descriptor, classifier, img):
        #img = np.array([utils.load_image(path)])
        feed_dict = {self.images:img}
        fc7 = self.sess.run(descriptor.relu7, feed_dict=feed_dict)

        pred = np.array(list(classifier.predict_proba(fc7, as_iterable=True)))
        res = []
        for ii in range(pred.shape[0]):
            ind = pred[ii].argmax()
            if pred[ii][ind] < 0.8:
                res.append(-1)
            else:
                res.append(ind)
        return res

    def recognise(self, coordinates, img_path):
        images = self.cropFaces(coordinates, img_path)
        indexes = self.getname(self.vgg, self.classifier, images)
        res = []
        for ii in range(len(coordinates)):
            if indexes[ii] < 0:
                res.append(("unknown", coordinates[ii]))
            else:
                res.append((self.uid[indexes[ii]], coordinates[ii]))    

        self.exportImage(res, img_path)
        return len(coordinates), res

    def exportImage(self, faces, img_path):
        img = cv2.imread(img_path)
        for (name, (x, y, w, h)) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(img_path, img)

    def cropFaces(self, faces, img_path):
        img = cv2.imread(img_path)
        imgs = np.zeros((len(faces), 224, 224, 3))
        count = 0
        for (x, y, w, h) in faces:
            imgs[count] = utils.normalize_image(cv2.cvtColor(img[y:y+h, x:x+w, :],
                                                             cv2.COLOR_BGR2RGB))
            count += 1
        return imgs
