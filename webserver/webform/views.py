from django.shortcuts import render

import os
import random
import string
import requests as rqs
import json
import cv2
from json2html import *
import config

def home(request):
	return render(request, 'index.htm', {})
 
def detect(request):
	if request.method == 'POST':
		if request.FILES.has_key('img_detect'):
			tmp_filename = 'media/' + ''.join(random.choice(string.ascii_uppercase 
				+ string.digits) for _ in range(64)) + '.jpg'
			handle_uploaded_file(request.FILES['img_detect'], tmp_filename)
			response = rqs.post(config.API_DETECT,
							 files={'image':open(tmp_filename, 'rb')}).text
			html_result = render_json_result(response)
			render_detected_image(response, tmp_filename)
			return render(request, 'detect.htm',
				{'uploaded_image':'/' + tmp_filename,
				 'json_result':html_result})	   
 
	return render(request, 'detect.htm', {})
 
def recognise(request):
	if request.method == 'POST':
		if request.FILES.has_key('img_recognise'):
			tmp_filename = 'media/' + ''.join(random.choice(string.ascii_uppercase 
				+ string.digits) for _ in range(64)) + '.jpg'
			handle_uploaded_file(request.FILES['img_recognise'], tmp_filename)

			response = rqs.post(config.API_RECOGNISE,
							 files={'image':open(tmp_filename, 'rb')}).text
			html_result = render_json_result(response)
			render_recognised_image(response, tmp_filename)
			return render(request, 'recognise.htm', 
				{'uploaded_image':'/' + tmp_filename,
				 'json_result':html_result})

	return render(request, 'recognise.htm', {})

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

def render_recognised_image(response, filename):
	jsonRslt = json.loads(response)
	if not jsonRslt.has_key('coordinates'):
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
