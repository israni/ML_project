import numpy as np
import csv 
import os
import sys
import matplotlib.pyplot as plt
csvFile = open("trainingData.csv", "r")
csv_reader = csv.reader(csvFile, delimiter=',')
labels = ['iteration','loss','policy entropy', 'policy_loss', 'value_loss','values','rewards', 'cumulative rewards \n per episode', 'steps per episode', 'mean rewards \n per episode']
items = [[] for i in range(len(labels))]
for row in csv_reader:
	for i in range(len(row)):
		items[i].append(float(row[i]))

def reject_outliers(data,m=2):
    return abs(data - np.mean(data)) < m * np.std(data)
filtered = np.where(reject_outliers(items[1]))
print(np.size(filtered))
for i in range(len(row)):
	items[i] = np.array(items[i])[filtered]


plt.rcParams.update({'font.size': 16})	
csvFile.close()
figsize = (150., 6.)
n = 6
for i in range(n):
	plt.figure(i,figsize = figsize)
	plt.plot(items[0], items[i+1], label = labels[i+1])
	plt.ylabel(labels[i+1], fontsize=20)
	plt.xlabel('number of steps', fontsize=20)


plt.show()
