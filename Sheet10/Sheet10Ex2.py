import numpy as np


R = 0.16
Nz = np.append(1,np.arange(10,210,10))
M = 10

def RMSE(weights, samples, N):
    return 1/np.sqrt(N)*(np.dot(samples, weights.T)**2).sum()

def EffSampleSize(weights):
    return 1/(weights**2).sum()

def likelihood(obs, z, R):
    term = [np.exp(-1/2*(obs - zi)**2/R) for zi in z]
    return np.array(term)

def KalUpdate(zf, obs, n, Nz, R):
    Pa = np.diag(np.append(1/(1+1/R)*np.ones(n), np.ones(Nz-n)))
    za = zf + 1/(R*(1+1/R))* obs* np.eye(1,Nz,n) #maybe n or n-1 --> check
    return za, Pa

for item in Nz:
    print(item)
    vars()["rootMSE_{}".format(item)] = 0
    for i in range(1000):
        vars()["w_{}".format(item)] = np.ones((M,1))/M
        za = np.zeros(item)
        Pa = np.diag(np.ones(item))
        z = np.random.multivariate_normal(za, Pa, size = M)
        for n in range(item):# number of iterations
            yobs = np.random.normal(loc = 0, scale = R)
            vars()["w_{}".format(item)] = (vars()["w_{}".format(item)]*likelihood(yobs, z, R))/(vars()["w_{}".format(item)]*likelihood(yobs, z, R)).sum()
            za, Pa = KalUpdate(za, yobs, n, item, R)
            print(max(vars()["w_{}".format(item)]))
        vars()["rootMSE_{}".format(item)] += RMSE(vars()["w_{}".format(item)], z, item)
    vars()["rootMSE_{}".format(item)] = vars()["rootMSE_{}".format(item)]/1000
    print(np.sqrt(vars()["rootMSE_{}".format(item)]))

