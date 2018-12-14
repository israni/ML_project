import argparse
import json

import matplotlib.pyplot as plt


def log_play(filename1, filename2, figsize=None, output=None):
    

    with open(filename1, 'r') as f1:
        data1 = json.load(f1)
    if 'episode' not in data1:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    nb_steps1 = data1['nb_steps']

    with open(filename2, 'r') as f2:
        data2 = json.load(f2)
    if 'episode' not in data2:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    nb_steps2 = data2['nb_steps']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys1 = sorted(list(set(data1.keys()).difference(set(['nb_steps','mean_absolute_error','mean_eps','nb_episode_steps']))))
    keys2 = sorted(list(set(data2.keys()).difference(set(['nb_steps','mean_absolute_error','mean_eps','nb_episode_steps']))))

    if figsize is None:
        figsize = (150., 5.)
    
    for i in range(0,len(keys1)):
    	plt.figure(figsize=figsize)
    	plt.plot(nb_steps1,data1[keys1[i]])
    	plt.plot(nb_steps1,data2[keys2[i]])
    	plt.xlabel('number of steps')
    	plt.ylabel(keys1[i])
    
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


parser = argparse.ArgumentParser()
parser.add_argument('filename1', type=str, help='The filename of the JSON log generated during training.')
parser.add_argument('filename2', type=str, help='The filename of the JSON log generated during training.')
#parser.add_argument('filename3', type=str, help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.
log_play(args.filename1,args.filename2, output=args.output, figsize=args.figsize)
