from scipy.io import arff
import matplotlib
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
matplotlib.use('Agg')


def print_satimage_plots(satimage, save):
    print(satimage)
    path = './plots/satimage/'

    # 'class' distribution
    attribute_counter = sorted(Counter(satimage['class']).items())
    labels, values = zip(*attribute_counter)
    indexes = np.arange(len(labels))
    width = 0.5
    color = ['green', 'blue', 'purple', 'gold', 'brown', 'black']

    plt.bar(indexes, values, width, color=color)
    plt.title("Class distribution")
    plt.xticks(indexes, labels)
    if save:
        plt.savefig(path + "Class_Dist.png")
    plt.close()

    # 'E11' distribution
    min_val = min(satimage['E11attr'])
    max_val = max(satimage['E11attr'])
    step = 0.25
    indexes = np.arange(start=min_val, stop=max_val, step=step)
    plt.hist(satimage['E11attr'], bins=indexes, color='darkgreen')
    plt.xticks(np.arange(start=min_val, stop=max_val, step=step * 2))
    plt.title("E11 distribution")
    if save:
        plt.savefig(path + "E11_Dist.png")
    plt.close()

    # 'F12' distribution
    min_val = min(satimage['F12attr'])
    max_val = max(satimage['F12attr'])
    step = 0.25
    indexes = np.arange(start=min_val, stop=max_val, step=step)
    plt.hist(satimage['F12attr'], bins=indexes, color='darkblue')
    plt.xticks(np.arange(start=min_val, stop=max_val, step=step * 2))
    plt.title("F12 distribution")
    if save:
        plt.savefig(path + "F12.png")
    plt.close()

    # 'A13' distribution
    min_val = min(satimage['A13attr'])
    max_val = max(satimage['A13attr'])
    step = 0.25
    indexes = np.arange(start=min_val, stop=max_val, step=step)
    plt.hist(satimage['A13attr'], bins=indexes, color='goldenrod')
    plt.xticks(np.arange(start=min_val, stop=max_val, step=step * 2))
    plt.title("A13 distribution")
    if save:
        plt.savefig(path + "A13_Dist.png")
    plt.close()

    # 'B14' distribution
    min_val = min(satimage['A13attr'])
    max_val = max(satimage['A13attr'])
    step = 0.25
    indexes = np.arange(start=min_val, stop=max_val, step=step)
    plt.hist(satimage['B14attr'], bins=indexes, color='indigo')
    plt.xticks(np.arange(start=min_val, stop=max_val, step=step * 2))
    plt.title("B14 distribution")
    if save:
        plt.savefig(path + "B14_Dist.png")
    plt.close()

    print('Done!')


def print_ozone_plots(ozone, save):
    print(ozone)

    path = './plots/ozone/'

    # 'class' distribution
    attribute_counter = sorted(Counter(ozone['Class']).items())
    labels, values = zip(*attribute_counter)
    indexes = np.arange(len(labels))
    width = 0.5
    color = ['blue', 'red']

    plt.bar(indexes, values, width, color=color)
    plt.title("Class distribution")
    plt.xticks(indexes, labels)
    if save:
        plt.savefig(path + "Class_Dist.png")
    plt.close()

    # V1-V72
    for i in range(1, 73):
        min_val = min(ozone['V' + str(i)])
        max_val = max(ozone['V' + str(i)])
        step = (max_val - min_val) / 15.0
        indexes = np.arange(start=min_val, stop=max_val + step, step=step)
        plt.hist(ozone['V' + str(i)], bins=indexes, color='grey')
        plt.xticks(np.arange(start=min_val, stop=max_val + step, step=step * 2))
        plt.title("V" + str(i) + " distribution")
        if save:
            plt.savefig(path + "V" + str(i) + "_Dist.png")
        plt.close()

    print("Done!")


if __name__ == '__main__':
    satimage_arff = arff.loadarff('./datasets/dataset_186_satimage.arff')
    ozone_arff = arff.loadarff('./datasets/phpdReP6S.arff')

    satimage = pd.DataFrame(satimage_arff[0])
    ozone = pd.DataFrame(ozone_arff[0])

    print_satimage_plots(satimage, False)
    print_ozone_plots(ozone, False)
