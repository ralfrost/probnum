import numpy as np
from matplotlib import pyplot as plt
from probnum import random_variables as rvs
from probnum.linalg.linops import Kronecker
import time

DIM_V = 15
DIM_W = 20
SIZE = (50,20)

#create matrixvariate normal distribution
V = np.random.rand(DIM_V, DIM_V)
V = np.dot(V, V.T)
W = np.random.rand(DIM_W, DIM_W)
W = np.dot(W,W.transpose())
#V = 100*np.diag(np.abs(np.random.rand(DIM_V)))
#W = 100*np.diag(np.abs(np.random.rand(DIM_W)))
x1=np.linspace(0,1,DIM_V)
x2=np.linspace(0,2,DIM_W)
print("Shapex1",x1.shape," shapex2",x2.shape)
mean = np.outer(x1,x2)
#mean = np.random.rand(DIM_V, DIM_W)
cov = Kronecker(V, W)
N = rvs.Normal(mean, cov)
print("mean = ", mean)
print("V = ", V)
print("W = ", W)
"""
if isinstance(SIZE, int):
    size_dim = 1
else:
    size_dim = len(SIZE)

method_names = ['sample_new_chol', 'sample_new_svd']
methods = [N.sample, N.sample]
params = [[SIZE],[SIZE],[SIZE]]
samples = []
err = []
sampmean = []

for i in range(2):
    np.random.seed(496)
    print("-----------------------------------------------\n>>>>> ",method_names[i],":")
    start = time.time()
    samples.append( methods[i](*params[i]))
    stop = time.time()
    print("Sampling time: ",stop-start," seconds.")
    print("Output shape = ",samples[i].shape)
    sampmean.append( np.mean(samples[i], axis = tuple(np.arange(size_dim))))    
    err.append( np.abs(mean-sampmean[i]) )
    print("Mean of error = ", np.mean(err[i]),", Sum of error = ",np.sum(err[i]))
    #print("Sampled mean: ",sampmean[i])
    #print("Error from mean = ",err[i])

#create graph
vmax = 1.5
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22.5, 7), sharey=True)
for j in range(2):
    matdict = {"$\mathbb{E}(\mathsf{X})$": mean, "$V$": V, "$W$": W, method_names[j]: sampmean[j], "absolute error": err[j]}
    for i, (title, rv) in enumerate(matdict.items()):
        axes[j][i].imshow(rv, cmap='bwr', vmin=-vmax, vmax=vmax)
        axes[j][i].set_axis_off()
        axes[j][i].title.set_text(title)

plt.tight_layout()
plt.savefig('samples_mean.png')
plt.show()

#TODO: create qq-plot to simultaneously test mean and covariance
"""