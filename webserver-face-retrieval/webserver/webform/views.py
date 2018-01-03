from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

import os
import requests as rqs
import json
import cv2
from json2html import *
import config
import glob
import numpy as np
import random
import skimage
import string

import sys
sys.path.insert(0, '/home/tmtriet/Desktop/AnNT/WebDemo-FaceRetrieval/SSDFace')
sys.path.insert(0, '/home/tmtriet/Desktop/AnNT/WebDemo-FaceRetrieval/vgg16')
import VggRecogniser
import FaceDetection

def home(request):
    return render(request, 'index.htm', {})
 
def detect(request):
    if request.method == 'POST':
        if request.FILES.has_key('img_detect'):
            filename = str(request.FILES['img_detect'])
            handle_uploaded_file(request.FILES['img_detect'], filename)
            response = rqs.post(config.API_DETECT,
                             files={'image':open('media/' + filename, 'rb')}).text
            print response
            html_result = render_json_result(response)
            render_detected_image(response, 'media/' + filename)
            return render(request, 'detect.htm',
                {'uploaded_image':'/media/' + filename,
                 'json_result':html_result})       
 
    return render(request, 'detect.htm', {})
 
def recognise(request):
    if request.method == 'POST':
        if request.FILES.has_key('img_recognise'):
            filename = str(request.FILES['img_recognise'])
            handle_uploaded_file(request.FILES['img_recognise'], filename)

            response = rqs.post(config.API_RECOGNISE,
                             files={'image':open('media/' + filename, 'rb')}).text
            print response
            html_result = render_json_result(response)
            render_recognised_image(response, 'media/' + filename)
            return render(request, 'recognise.htm', 
                {'uploaded_image':'/media/' + filename,
                 'json_result':html_result})

    return render(request, 'recognise.htm', {})

def retrieval(request):
    if request.method == 'POST':
        if request.FILES.has_key('img_retrieval'):
            #filename = str(request.FILES['img_retrieval'])
            filename = 'media/' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(64)) + '.jpg'
            handle_uploaded_file(request.FILES['img_retrieval'], filename)
            
            '''
            response = rqs.post(config.API_RECOGNISE,
                             files={'image':open('media/' + filename, 'rb')}).text
            '''
            res_detect = SSD_detect(filename)
            res_recognise = VGG_recognise(res_detect, filename)

            html_result = render_json_result(json.dumps(res_recognise))
            render_recognised_image(json.dumps(res_recognise), filename)
            re_result = render_retrieval_html(json.dumps(res_recognise))
            return render(request, 'retrieval.htm', 
                {'uploaded_image': '/'+filename,
                 'json_result':html_result,
                 'retrieval_results':re_result})


    return render(request, 'retrieval.htm', {})

def SSD_detect(filename):
    try:
        img = skimage.io.imread(filename)
        _ssd_fd = FaceDetection.FaceDetection()
        cc, scores, bboxes = _ssd_fd.dectectFace(img)
        res = {}
        if len(scores) < 1:
            res = {'code':config.CODE_NON_FACE}
        else:
            #visualization.bboxes_draw_on_img(img, cc, scores, bboxes, visualization.colors, class_names=['none-face', 'face'])
            #skimage.io.imsave(tmp_filename, img)
            bboxes = normalizeBBoxes(bboxes, img.shape[1], img.shape[0])

            res['code'] = config.CODE_SUCCESS
            res['num'] = len(scores)
            res['coordinates'] = []
            for ii in range(len(scores)):
                if scores[ii] >= config.SCORE_THRES:
                    y1, x1, y2, x2 = bboxes[ii]
                    res['coordinates'].append("%d,%d,%d,%d"%(x1, y1, x2 - x1, y2 - y1))
            res["url"] = filename
    except:
        res = {'code':config.CODE_SYS_ERR}
    return res

def normalizeBBoxes(bboxes, w, h):
    bboxes[:, [0, 2]] *= h
    bboxes[:, [1, 3]] *= w
    bboxes = np.floor(bboxes).astype(int)
    return bboxes

def VGG_recognise(jsonRspn, filename):
    try:
        if jsonRspn['code'] == config.CODE_SUCCESS:
            coordinates = jsonRspn["coordinates"]
            bboxes = []
            for it in coordinates:
                x, y, w, h = it.split(',')
                bboxes.append((int(x), int(y), int(w), int(h)))

        if len(bboxes) > 0:
            _vgg16_classifier = VggRecogniser.VggRecogniser()
            num, faces = _vgg16_classifier.recognise(bboxes, filename)

            res = {}
            res['num'] = num
            if num < 1:
                res['code'] = config.CODE_NON_FACE
            else:
                res['code'] = config.CODE_SUCCESS
                res['names'] = []
                res['coordinates'] = []
                for (name, (x, y, w, h)) in faces:
                    res['names'].append(name)
                    res['coordinates'].append('%d,%d,%d,%d'%(x, y, w, h))
                res['url'] = filename
    except:
        res = {'code':config.CODE_SYS_ERR}
    return res

def handle_uploaded_file(file, filename):
    if not os.path.exists('media/'):
        os.mkdir('media/')
 
    with open(filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

def render_detected_image(response, filename):
    coordinates = json.loads(response)['coordinates']
    img = cv2.imread(filename)
    for it in coordinates:
        x,y,w,h = it.split(',')
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(filename, img)

def render_retrieval_html(response):
    jsonRslt = json.loads(response)
    if int(jsonRslt['num']) == 0:
        return ""
    names = jsonRslt['names']
    res = ""
    for it in names:
        print '%s/%s/*.jpg'%(config.DATA_DIR, it)
        files = glob.glob('%s/%s/*.jpg'%(config.DATA_DIR, it))[:config.NUM_RETURN]
        print files
        for ii in range(len(files)):
            res += '<div class=\"gallery\"><img src=\"%s\" max-width=\"300\" max-height=\"200\"></div>'%(
                files[ii].replace(config.DATA_DIR, '/static/data'))

    return res

def render_recognised_image(response, filename):
    jsonRslt = json.loads(response)
    if int(jsonRslt['num']) == 0:
    	return
    coordinates = jsonRslt['coordinates']
    names = jsonRslt['names']

    img = cv2.imread(filename)
    num = len(coordinates)
    for ii in range(num):
        x,y,w,h = coordinates[ii].split(',')
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, names[ii],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(filename, img)

def render_json_result(response):
    return json2html.convert(json = response)