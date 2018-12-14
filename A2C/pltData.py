import numpy as np
import csv 
import os
import sys
import matplotlib.pyplot as plt
csvFile = open("trainingData.csv", "r")
csv_reader = csv.reader(csvFile, delimiter=',')
labels = ['iteration','loss','policy entropy', 'policy_loss', 'value_loss','values','rewards', 'cumulative rewards \n per episode', 'steps per episode', 'mean rewards \n per episode']
print(len(labels))
items = [[] for i in range(len(labels))]
for row in csv_reader:
	for i in range(len(row)):
		items[i].append(float(row[i]))

	
csvFile.close()
plotNo = 511
plt.figure(0)
ax1 = plt.subplot(plotNo)
n = int(len(items)/2)
for i in range(n):
	print(i)
	ax = plt.subplot(plotNo, sharex=ax1)
	plt.plot(items[0], items[i+1], label = labels[i+1])
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.ylabel(labels[i+1])
	plotNo += 1
plt.setp(ax.get_xticklabels(), visible=True)
plt.xlabel('iteration')
plt.figure(1)
plotNo = 411
ax1 = plt.subplot(plotNo)
for i in range(n-1):
	print(n+i+1)
	ax = plt.subplot(plotNo, sharex=ax1)
	plt.plot(items[0], items[n+i+1], label = labels[n+i+1])
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.ylabel(labels[n+i+1])
	plotNo += 1
plt.setp(ax.get_xticklabels(), visible=True)
plt.xlabel('iteration')

plt.show()
