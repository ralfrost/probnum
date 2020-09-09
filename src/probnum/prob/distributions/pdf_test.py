import numpy as np
from normal import Normal
from probnum.linalg.linops import Kronecker
from scipy.stats import multivariate_normal
import pylab
import time

DIM_V = 5
DIM_W = 7
np.random.seed(4002)
#create matrixvariate normal distribution
V = np.random.rand(DIM_V,DIM_V)
V = np.dot(V,V.T)+5*np.eye(DIM_V)
print("cond V", np.linalg.cond(V))
W = np.random.rand(DIM_W,DIM_W)
W = np.dot(W,W.T)+5*np.eye(DIM_W)
print("cond W", np.linalg.cond(W))

mean = np.random.rand(DIM_V,DIM_W)
'''
V=np.array([[2,-1],[-1,1]])
W=np.array([[14,14,17],[14,17,6],[17,6,61]])
mean = np.array([[2,1,0],[-1,0,1]])
'''
cov = Kronecker(V,W)
N = Normal(mean, cov)
print("mean ",mean)
#generate X and calculate pdf(X) + time measurement
#X = N.sample(((),1))
X = mean

start = time.time()
pdf_new_chol = N.pdf(X, 'cholesky')
stop = time.time()
print("New function took with cholesky ",stop-start,"seconds.")

start = time.time()
pdf_new_svd = N.pdf(X, 'svd')
stop = time.time()
print("New function took with svd ",stop-start,"seconds.")

#calculate pdf(X) using multivariate methods
start = time.time()
mean = mean.ravel()
#print(">>>MEAN = ",mean)
X = X.ravel()
#print(">>>X = ",X)
cov_dense = cov.todense()
#print("shape mean",mean.shape,", shape cov = ",cov_dense.shape)
#print("cov = ",cov_dense)
#print("cond cov", np.linalg.cond(cov_dense))
try: 
    inv = np.linalg.inv(cov_dense)
except:
    print("!!!! COV NOT INVERTIBLE !!!!")
dist = multivariate_normal(mean, cov_dense)
pdf_X_old = dist.pdf(X)
stop = time.time()
print("Old function took ",stop-start,"seconds.")

print("New cholesky pdf(X) = ",pdf_new_chol)
print("New svd pdf(X) = ",pdf_new_svd)
print("Old pdf(X) = ",pdf_X_old)
print("relative error = ",np.abs((np.log(pdf_new_chol)-np.log(pdf_X_old))/np.log(pdf_X_old)))
print("Difference = ",pdf_new_chol-pdf_X_old)



'''
    def classic_pdf(self, x, method='cholesky'):
        (m,n) = self.mean().shape
        cov_dense = self.cov().todense()
        dev = x - self.mean()
        dev = dev.T.ravel()
        cov_inv = np.linalg.inv(cov_dense)
        cov_det = np.linalg.det(cov_dense)
        print("classic logabsdet = ",np.log(cov_det))
        maha = dev @ cov_inv @ dev
        return np.exp( -0.5*(m*n*np.log(2*np.pi) + np.log(cov_det) + maha))
'''