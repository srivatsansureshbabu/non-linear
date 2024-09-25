import sys
import struct
import numpy as np

def main():
	if len(sys.argv) < 2:
		print("Specify a file name")
		return

	nFeatures, nClasses, X, y = readChoirDat(sys.argv[1])

	print("Read dataset in X & Y (list type)")
	print("# of features: %d" % nFeatures)
	print("# of classes: %d" % nClasses)
	print("# of data points: %d" % len(X))

def readChoirDat(filename):
	""" Parse a choir_dat file """
	with open(filename, 'rb') as f:
		nFeatures = struct.unpack('i', f.read(4))[0]
		nClasses = struct.unpack('i', f.read(4))[0]
		
		X = []
		y = []

		while True:
			newDP = []
			for i in range(nFeatures):
				v_in_bytes = f.read(4)
				if v_in_bytes is None or len(v_in_bytes) == 0:
					return nFeatures, nClasses, X, y

				v = struct.unpack('f', v_in_bytes)[0]
				newDP.append(v)

			l = struct.unpack('i', f.read(4))[0]
			X.append(newDP)
			y.append(l)

	return nFeatures, nClasses, X, y

# See this - if you wan to generate using this format
def writeDataSetForChoirSIM(ds, filename):
	X, y = ds

	f = open(filename, "wb")

	nFeatures = len(X[0])
	nClasses = len(set(y))

	f.write(struct.pack('i', nFeatures))
	f.write(struct.pack('i', nClasses))
	for V, l in zip(X, y):
		for v in V:
			f.write(struct.pack('f', v))
		f.write(struct.pack('i', l))

if __name__ == "__main__":
	nFeat, nClass, data, labels = readChoirDat('../dataset/emotion/emotion2_train.choir_dat')
	print(nFeat, nClass, len(data), len(labels))
	print(set(labels))
	for i in range(len(data[0])):
		print(data[0][i])
	exit()
	counts = np.zeros(2)
	for i in range(len(data)):
		counts[labels[i]] += 1
	print(counts)
	#print(len(newData))

	#ds = (newData, newLabels)
	#writeDataSetForChoirSIM(ds, '../dataset/emotion/emotion2_test.choir_dat')

	


