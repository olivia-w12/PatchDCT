import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_cdf(datas,name='CDF.jpg'):
    nt, bins, patches = plt.hist(datas, bins=1000, histtype='step',
                                 cumulative=True, density=True, color='darkcyan')
    plt.title('bins = 1000')
    plt.savefig(name, dpi=200)
    plt.show()
    plt.close()

datas = np.array([64.3, 65.0, 65.0, 67.2, 67.3, 67.3, 67.3, 67.3, 68.0, 68.0, 68.8, 68.8, 68.8, 69.7,\
                  69.7, 69.7, 70.3,70.4, 70.4, 70.4, 70.4, 70.4,70.4, 70.4, 71.2, 71.2, 71.2, 71.2,\
                  72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.7, 72.7, 72.7, 72.7, 72.7, 72.7, 72.7,\
                  73.5, 73.5, 73.5, 73.5, 73.5, 73.5, 73.5, 73.5, 73.5,73.5, 73.5, 74.3, 74.3, 74.3,\
                  74.3, 74.3, 74.3, 74.3, 74.3, 74.7, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.4,\
                  75.6, 75.8, 75.8, 75.8, 75.8, 75.8, 76.5, 76.5, 76.5, 76.5, 76.5, 76.5, 76.5, 77.2,\
                  77.2,77.6, 78.0, 78.8, 78.8, 78.8, 79.5, 79.5, 79.5, 80.3, 80.5, 80.5, 81.2, 81.6,\
                  81.6, 84.3])
datas = torch.from_numpy(datas)
histc = torch.histc(datas,100)
histogram = torch.histogram(datas,density=True)
print('----------------------')
#add a test for git
