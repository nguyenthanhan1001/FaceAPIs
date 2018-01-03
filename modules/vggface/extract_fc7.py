import VggRecogniser as VR
import os
import numpy as np

fr = VR.VggRecogniser()
f = open('/home/mmhci_hcmus/data/faces-5vd/faces.txt', 'rt')
count = 0
for path in f:
	path = path.strip()
	print path
	fc7 = fr.extract_fc7(path)
	name = path.replace('/faces-5vd/', '/feats-5vd/')
	if not os.path.exists(os.path.dirname(name)):
		os.makedirs(os.path.dirname(name))
	np.save(name, fc7)
	count += 1
	print count
f.close()