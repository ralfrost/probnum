import numpy as np
from normal import Normal
from probnum.linalg.linops import Kronecker


mean = np.arange(6).reshape(3,2)
print("mean = ",mean)
A = np.array([[1,0,1],[2,1,0],[-1,1,0]])
V = np.dot(A,A.T)
B = np.array([[1,0],[1,-1]])
W = np.dot(B,B.T)

cov = Kronecker(V,W)
N = Normal(mean, cov)
print("cov type = ",type(cov.A.A))
sample = N.sample(4)
X1 = np.array([[0.8,1.8],[3.2,3.8]])
X2 = np.array([[0.5,2.5],[3.5,3.5]])

print("sample = ",sample)
print("X1 = ",X1,"pdf(x) = ",N.pdf(X1))                             
print("X2 = ",X2,"pdf(x) = ",N.pdf(X2))                             
