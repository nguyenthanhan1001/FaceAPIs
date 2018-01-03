import glob
import os
import sys
import random

def write(filename, data, lbl):
	f = open(filename, 'wt')
	for i in range(len(data)):
		for k in range(len(data[i])):
			f.write(os.path.abspath(data[i][k])
				+ '\t%d\n'%(lbl[i][k]))
	f.close()

	f = open(filename, 'rt')
	lines = f.readlines()
	f.close()
	f = open(filename, 'wt')
	random.shuffle(lines)
	f.writelines(lines)
	f.close()

if __name__ == "__main__":
	in_dir = sys.argv[1]
	fds = glob.glob(in_dir + '*/')
	#print fds
	
	train = []
	lbltrain = []
	lblval = []
	val = []
	test = []
	lbltest = []
	for it in fds:
		pics = glob.glob(it + '*')
		num = len(pics)
		indx = int(os.path.basename(os.path.dirname(pics[0])))

		ntrain = int(num * 0.9)
		nval = int(num * 0.0)
		ntest = num - ntrain - nval

		print num, ntrain, nval, ntest
		train.append(pics[:ntrain])
		val.append(pics[ntrain:ntrain+nval])
		test.append(pics[ntrain+nval:])

		tmp = []
		for i in range(ntrain):
			tmp.append(indx)
		lbltrain.append(tmp)
		tmp = []
		for i in range(nval):
			tmp.append(indx)
		lblval.append(tmp)
		tmp = []
		for i in range(ntest):
			tmp.append(indx)
		lbltest.append(tmp)

	print len(train), len(val), len(test)

	write('train.txt', train, lbltrain)
	write('val.txt', val, lblval)
	write('test.txt', test, lbltest)
