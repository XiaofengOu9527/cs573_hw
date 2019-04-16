import numpy as np
import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
from kmeans import kmeans

np.random.seed(0)

arg = sys.argv
dataFilenames = []
dataFilenames.append(arg[1])
dataFilenames.append(arg[2])
dataFilenames.append(arg[3])


nmi = np.zeros(len(dataFilenames))
Ks = (8, 8, 2)

for i in range(len(dataFilenames)):
    dataFilename = dataFilenames[i]
    K = Ks[i]
    data = pd.read_csv(dataFilename, header=None)
    N = data.shape[0]

    _, _, nmi[i], predict_label = kmeans(dataFilename, K)

    print('NMI of dataset{} with K={}: {}'.format(str(i+1), K, nmi[i]))

    sample_idxs = np.random.randint(0, N, size=1000)
    sample_imgs = data.iloc[sample_idxs, 2:].to_numpy()
    sample_labels = predict_label[sample_idxs]

    df = pd.DataFrame(dict(x=sample_imgs[:, 0], y=sample_imgs[:, 1],        label=sample_labels))
    groups = df.groupby('label')

    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.scatter(group.x, group.y, label=name)
    ax.legend()
    plt.savefig("visualize_dataset{}_with_K={}".format(str(i+1), K))



