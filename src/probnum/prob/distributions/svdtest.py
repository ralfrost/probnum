import numpy as np                                                                                   
from scipy import linalg                                                                            
from scipy.sparse import linalg as slinalg                                                           

x = np.array([[1,1,0,0,0],[0,0,1,1,0],[1,1,1,1,1]],dtype=np.float64)                                 
print("shape = ", x.shape)
npsvd = np.linalg.svd(x)                                                                             
spsvd = linalg.svd(x)                                                                                
sptop = slinalg.svds(x,k=2)                                                                          

print( "np"           )                                                                                
print( "u: ", npsvd[0])                                                                                
print( "s: ", npsvd[1])                                                                                
print( "v: ", npsvd[2])                                                                                

print( "\n=========================\n" )                                                               

print( "sp"                            )                                                               
print( "u: ", spsvd[0]                 )                                                               
print( "s: ", spsvd[1]                 )                                                               
print( "v: ", spsvd[2]                 )                                                               

print( "\n=========================\n" )                                                               

print( "sp top k"                      )                                                               
print( "u: ", sptop[0]                 )                                                               
print( "s: ", sptop[1]                 )                                                               
print( "v: ", sptop[2]                 )                                                               

nptmp = np.zeros((npsvd[0].shape[1],npsvd[2].shape[0]))                                              
nptmp[np.diag_indices(np.min(nptmp.shape))] = npsvd[1]                                               
npreconstruct = np.dot(npsvd[0], np.dot(nptmp,npsvd[2]))                                             

print( npreconstruct                                )                                                  
print( "np close? : ", np.allclose(npreconstruct, x))                                                  

sptmp = np.zeros((spsvd[0].shape[1],spsvd[2].shape[0]))                                              
sptmp[np.diag_indices(np.min(sptmp.shape))] = spsvd[1]                                               
spreconstruct = np.dot(spsvd[0], np.dot(sptmp,spsvd[2]))                                             

print( spreconstruct  )                                               
print( "sp close? : ", np.allclose(spreconstruct, x))

sptoptmp = np.zeros((sptop[0].shape[1],sptop[2].shape[0]))                                           
sptoptmp[np.diag_indices(np.min(sptoptmp.shape))] = sptop[1]                                         
sptopreconstruct = np.dot(sptop[0], np.dot(sptoptmp,sptop[2]))                                       

print( sptopreconstruct      ) 
print( "sp top close? : ", np.allclose(sptopreconstruct, x))