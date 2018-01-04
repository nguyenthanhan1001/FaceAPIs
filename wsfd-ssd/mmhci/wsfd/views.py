from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import random
import string

import skimage.io
import numpy as np
import urllib

import config

import sys
sys.path.insert(0, config.SSD300_DIR)
import FaceDetection
import visualization

global gb_detector
gb_detector = FaceDetection.FaceDetection()

def hello(request):
    return HttpResponse('mmHCI Face Detection Webservice')

def handle_uploaded_file(f, filename):
    with open(filename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def detect(request):
    tmp_filename = 'media/' + ''.join(random.choice(string.ascii_uppercase 
                        + string.digits) for _ in range(64)) + '.jpg'
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['image'], tmp_filename)
    elif request.method == 'GET':
        url = request.GET['url']
        try:
            urllib.urlretrieve(url, tmp_filename)
        except:
            return JsonResponse({'code':config.CODE_INV_URL})
    else:
        return JsonResponse({'code':config.CODE_MTHD_NOT_SPRT})

    try:
        img = skimage.io.imread(tmp_filename)
        global gb_detector
        cc, scores, bboxes = gb_detector.dectectFace(img)
        res = {}

        if len(scores) > 0:
            visualization.bboxes_draw_on_img(img, cc, scores, bboxes, 
                visualization.colors, class_names=['none-face', 'face'])
            skimage.io.imsave(tmp_filename, img)
            bboxes = normalizeBBoxes(bboxes, img.shape[1], img.shape[0])

        res['code'] = config.CODE_SUCCESS
        res['coordinates'] = []
        for ii in range(len(scores)):
            if scores[ii] >= config.SCORE_THRES:
                y1, x1, y2, x2 = bboxes[ii]
                res['coordinates'].append("%d,%d,%d,%d"%(x1, y1, x2 - x1, y2 - y1))
        res['num'] = len(res['coordinates'])
        res["url"] = '/' + tmp_filename
    except:
        res = {'code':config.CODE_SYS_ERR}
    return JsonResponse(res)

def normalizeBBoxes(bboxes, w, h):
    bboxes[:, [0, 2]] *= h
    bboxes[:, [1, 3]] *= w
    bboxes = np.floor(bboxes).astype(int)
    return bboxes