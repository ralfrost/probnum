import numpy as np
import scipy.stats as stats
import pylab
import probnum
from probnum import random_variables as rvs
from probnum.linalg import linops
from matplotlib import pyplot as plt
np.random.seed(496)
SIZE = 100000
n = 3
#generate covariance
A = np.random.uniform(size=(n, n))
#A = np.linspace(0.5,1.5,3)
#A = np.outer(A,A.T)
A =(A @ A.T)
#create distribution
rv = rvs.Normal(mean=np.eye(A.shape[0]), cov=linops.SymmetricKronecker(A=A), random_state=1)
samples_presymm = rv._symmetric_kronecker_identical_factors_sample2(size=SIZE)
samples_postsymm = rv.sample(size=SIZE)

#compute covariances 
samples_multivar = np.array([mat.ravel() for mat in samples_presymm]).T
print("shape = ", samples_multivar.shape)
sampledpresymm_cov = np.cov(samples_multivar)
original_cov = linops.SymmetricKronecker(A).todense()
print("Original COV: ", original_cov, "shape = ", original_cov.shape) 
print("Rel Error = ",np.abs(sampledpresymm_cov-original_cov)/np.abs(original_cov))

samples_multivar = np.array([mat.ravel() for mat in samples_postsymm]).T
print("shape = ", samples_multivar.shape)
sampledpostsymm_cov = np.cov(samples_multivar)
print("Original COV: ", original_cov, "shape = ", original_cov.shape) 
print("Rel Error = ",np.abs(sampledpostsymm_cov-original_cov)/np.abs(original_cov))
#create plot
vmax =np.max(original_cov.ravel()) 
print("vmax = ",vmax)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 7), sharey=True)
axes[0][0].imshow(np.tril(sampledpresymm_cov), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[0][0].set_axis_off()
axes[0][0].title.set_text("presymmetrized sampled COV")
axes[0][1].imshow(np.tril(original_cov), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[0][1].set_axis_off()
axes[0][1].title.set_text("original COV")
axes[0][2].imshow(np.tril(np.abs(sampledpresymm_cov-original_cov)), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[0][2].set_axis_off()
axes[0][2].title.set_text("Error")
axes[1][0].imshow(np.tril(sampledpostsymm_cov), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[1][0].set_axis_off()
axes[1][0].title.set_text("postsymmetrized sampled COV")
axes[1][1].imshow(np.tril(original_cov), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[1][1].set_axis_off()
axes[1][1].title.set_text("original COV")
axes[1][2].imshow(np.tril(np.abs(sampledpostsymm_cov-original_cov)), cmap='bwr', vmin=-vmax, vmax=vmax)
axes[1][2].set_axis_off()
axes[1][2].title.set_text("Error")
plt.tight_layout()
plt.show()
#create qq-plot
"""
for i in range(3):
    for j in range(3):
        x = (samples_presymm[:,i,j]-np.eye(3)[i,j])/np.sqrt(linops.SymmetricKronecker(A).todense()[i*j,i*j])
        print("i = ",i,", j = ",j)
        stats.probplot(x, dist="norm", plot=pylab)
        pylab.show()
        """