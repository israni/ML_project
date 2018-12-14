import argparse
import json

import matplotlib.pyplot as plt


def log_play(filename1, filename2, filename3):
    
	# file1 - data1
    with open(filename1, 'r') as f1:
        data1 = json.load(f1)
    nb_steps1 = data1['nb_steps']

    # file2 - data2
    with open(filename2, 'r') as f2:
        data2 = json.load(f2)
    nb_steps2 = data2['nb_steps']

    # file3 - data3
    with open(filename3, 'r') as f3:
        data3 = json.load(f3)
    nb_steps3 = data3['nb_steps']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data1.keys()).difference(set(['nb_steps','mean_absolute_error','mean_eps','nb_episode_steps']))))
    
    figsize = (150., 5.)
    

    for i in range(0,len(keys)):
    	#plt.figure(figsize=figsize)
    	fig, ax = plt.subplots(figsize = figsize)
    	ax.plot(nb_steps1[0:4676],data1[keys[i]][0:4676],label ='seed 111')
    	ax.plot(nb_steps2,data2[keys[i]],label = 'seed 555')
    	ax.plot(nb_steps3[0:4676],data3[keys[i]][0:4676],label = 'seed 999')
    	ax.legend()
    	plt.xlabel('number of steps')
    	plt.ylabel(keys[i])
    
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('filename1', type=str)
parser.add_argument('filename2', type=str)
parser.add_argument('filename3', type=str)
args = parser.parse_args()

# Call plotting function
log_play(args.filename1,args.filename2,args.filename3)
