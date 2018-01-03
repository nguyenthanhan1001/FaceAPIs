import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python import learn
import random
import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

NUM_PID = 69
FEATURE_SIZE = 4096
BATCH_SIZE = 1940

def next_batch(f, num):
	batch_x = np.zeros((num, FEATURE_SIZE), np.float32)
	batch_y = np.zeros((num)).astype(np.int32)
	count = 0
	while count < num:
		line = f.readline()
		if len(line) < 1:
			break
		path, lbl = line.strip().split('\t')
		batch_x[count] = np.load(path)
		batch_y[count] = int(lbl)
		count += 1
	return batch_x[:count], batch_y[:count], count
	
def shuffleLines(filename):
	f = open(filename, 'rt')
	lines = f.readlines()
	f.close()
	random.shuffle(lines)
	f = open(filename, 'wt')
	f.writelines(lines)
	f.close()

def testAcc(classifier, ftest):
	count = 0
	correct = 0
	ftest.seek(0, 0)
	while True:
		batch_x, batch_y, num = next_batch(ftest, BATCH_SIZE)
		lbl = np.array(list(classifier.predict(batch_x)))
		grntrth = batch_y.argmax(0)
		correct += sum(grntrth == lbl)
		count += num
		if num < BATCH_SIZE:
			break
	return float(correct) / count

def traceAcc(trace_Acc, curAcc, length):
	for i in range(length - 1):
		trace_Acc[i] = trace_Acc[i + 1]
	trace_Acc[length - 1] = curAcc
	count = 0
	for i in range(length - 1):
		if trace_Acc[i] > trace_Acc[i + 1]:
			count += 1
	return count == length - 1

def initGraph(model_dir, lr=1e-4):
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

	classifier = learn.DNNClassifier(
	    model_dir=model_dir,
	    feature_columns=feature_columns,
	    hidden_units=[4096, 2048],
	    n_classes=NUM_PID,
	    activation_fn=tf.nn.sigmoid,
	    optimizer=tf.train.AdamOptimizer(learning_rate=lr)

	)
	return classifier

if __name__ == "__main__":
	meta_dir = sys.argv[1]
	model_dir = sys.argv[2]

	if not os.path.exists(meta_dir + '/train.txt'):
		print 'Can not find train.txt in ' + meta_dir
	ftrain = open(meta_dir + '/train.txt', 'rt')

	'''
	if not os.path.exists(meta_dir + '/val.txt'):
		print 'Can not find val.txt in ' + meta_dir
	fval = open(meta_dir + '/val.txt', 'rt')
	'''

	if not os.path.exists(meta_dir + '/test.txt'):
		print 'Can not find test.txt in ' + meta_dir
	ftest = open(meta_dir + '/test.txt', 'rt')

	classifier = initGraph(model_dir=model_dir)
	print 'initted graph...'

	print 'training...'

	batch_x, batch_y, num = next_batch(ftest, BATCH_SIZE)
	print classifier.evaluate(x=batch_x, y=batch_y)['accuracy']

	'''
	num = 0
	trace_Acc = -np.ones(4)
	it = 0
	while True:
		batch_x, batch_y, num = next_batch(ftrain, BATCH_SIZE)
		if num > 0:
			#classifier.partial_fit(x=batch_x, y=batch_y)
			classifier.fit(x=batch_x, y=batch_y, max_steps=10000)
		if num < BATCH_SIZE:
			print 'cur-testing...'
			curAcc = testAcc(classifier, ftest)
			print curAcc
			flog = open('log.txt', 'at')
			flog.write('%05d\t%g\n'%(it, curAcc))
			flog.close()
			if traceAcc(trace_Acc, curAcc, 4):
				print 'completed...'
				break			
			shuffleLines(meta_dir + '/train.txt')
			ftrain = open(meta_dir + '/train.txt', 'rt')
			it += 1
			break
		print '.',

	'''
	ftrain.close()
	#fval.close()
	ftest.close()