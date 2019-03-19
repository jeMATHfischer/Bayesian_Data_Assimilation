import numpy as np

def likilihood(z):
    return np.exp(-(1-z)**2/2)

def normalizer(L,v):
    return 1/np.dot(L,v).sum()

def resampler(M, pi):
    p_resampled = np.zeros(len(pi))
    for i in range(M):
        dummy = np.zeros(len(pi))
        u = np.random.rand()
        ind = (pi.cumsum() > u).sum()
        dummy[ind] = 1
        p_resampled += dummy
    return p_resampled/p_resampled.sum()

z = np.array([1,2,3])

P = np.array([[1/2,1/4,1/4],[1/4,1/2,1/4],[1/4,1/4,1/2]])
p0 = np.array([0,0,1])
L = np.diag(likilihood(z))

def exact_filter(P, L, p):
    return np.dot(L,np.dot(P,p))

def sequential_MC(P, L, M, p):
    return np.dot(L, resampler(M,np.dot(P, p)))


M = [10,100,1000]


p_seq = p0

for i in range(100):
    p_seq = sequential_MC(P,L*normalizer(L,p_seq), M[0], p_seq)
