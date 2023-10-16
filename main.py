from scipy.io import arff
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np


def print_satimage_plots(satimage):
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
    plt.savefig(path + "Class_Dist.png")
    plt.show()

    # 'E11' distribution
    indexes = np.arange(start=-2.5, stop=3.0, step=0.25)
    plt.hist(satimage['E11attr'], bins=indexes, color='darkgreen')
    plt.xticks(np.arange(start=-2.5, stop=3.0, step=0.5))
    plt.title("E11 distribution")
    plt.savefig(path + "E11_Dist.png")
    plt.show()

    # 'F12' distribution
    indexes = np.arange(start=-2.5, stop=2.5, step=0.25)
    plt.hist(satimage['F12attr'], bins=indexes, color='darkblue')
    plt.xticks(np.arange(start=-2.5, stop=2.5, step=0.5))
    plt.title("F12 distribution")
    plt.savefig(path + "F12.png")
    plt.show()

    # 'A13' distribution
    indexes = np.arange(start=-3, stop=3, step=0.25)
    plt.hist(satimage['A13attr'], bins=indexes, color='goldenrod')
    plt.xticks(np.arange(start=-3, stop=3, step=0.5))
    plt.title("A13 distribution")
    plt.savefig(path + "A13_Dist.png")
    plt.show()

    # 'B14' distribution
    indexes = np.arange(start=-3, stop=4.5, step=0.25)
    plt.hist(satimage['B14attr'], bins=indexes, color='indigo')
    plt.xticks(np.arange(start=-3, stop=4.5, step=0.5))
    plt.title("B14 distribution")
    plt.savefig(path + "B14_Dist.png")
    plt.show()

    print(satimage)


if __name__ == '__main__':
    satimage_arff = arff.loadarff('./datasets/dataset_186_satimage.arff')
    ozone_arff = arff.loadarff('./datasets/phpdReP6S.arff')

    satimage = pd.DataFrame(satimage_arff[0])
    ozone_arff = pd.DataFrame(ozone_arff[0])

    print_satimage_plots(satimage)
