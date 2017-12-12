from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

import random
import string

import config
import requests as rqs
import json
import urllib
import sys
sys.path.insert(0, config.VGG16_DIR)
import VggRecogniser
import FaceDetector

global gb_classifier
gb_classifier = VggRecogniser.VggRecogniser()

def hello(request):
	return HttpResponse('mmHCI Face Recognition Webservice')

def handle_uploaded_file(f, filename):
	with open(filename, 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

@csrf_exempt
def recognise(request):
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

	res = {}
	try:
		num, faces = vggrecognise(tmp_filename)
		print num, faces
		res['num'] = num
		res['code'] = config.CODE_SUCCESS
		res['names'] = []
		res['coordinates'] = []
		for (name, (x, y, w, h)) in faces:
			res['names'].append(name)
			res['coordinates'].append('%d,%d,%d,%d'%(x, y, w, h))
		res['url'] = '/' + tmp_filename
	except:
		res = {'code':config.CODE_SYS_ERR}
	return JsonResponse(res)

def vggrecognise(tmp_filename):
	bboxes = ssd_detect(tmp_filename)
	if len(bboxes) > 0:
		global gb_classifier
		return gb_classifier.recognise(bboxes, tmp_filename)
	return 0, []

def ssd_detect(img_path):
	detectRspn = rqs.post(config.SSD_API, files={'image':open(img_path, 'rb')}).text
	jsonRspn = json.loads(detectRspn)
	if jsonRspn['code'] == config.CODE_SUCCESS:
		coordinates = jsonRspn["coordinates"]
		res = []
		for it in coordinates:
			x, y, w, h = it.split(',')
			res.append((int(x), int(y), int(w), int(h)))
		return res
	return []

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
		faces = FaceDetector.detectFace(tmp_filename)

		res = {}
		res['code'] = config.CODE_SUCCESS
		res['num'] = len(faces)
		res['coordinates'] = []
		for (x, y, w, h) in faces:
			res['coordinates'].append("%d,%d,%d,%d"%(x, y, w, h))
		res["url"] = '/' + tmp_filename
	except:
		res = {'code':config.CODE_SYS_ERR}
	return JsonResponse(res)