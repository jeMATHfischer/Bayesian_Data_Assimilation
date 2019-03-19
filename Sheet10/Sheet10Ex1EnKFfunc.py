import numpy as np
import matplotlib.pyplot as plt

def RMSEcalc(M, phi, o):
    P = 2
    R = 1

    Nobs = 100  # Nout = 1?
    Nout = 1
    # H = 1
    z = np.random.normal(loc=0, scale=P, size=(int(M[o]), 1))
    for p in range(len(phi)):
        vars()['z_{}'.format(p)] = np.random.normal(loc = 0, scale = P, size = (int(M[o]),1))
        vars()['rmse_{}'.format(p)] = 0
        vars()['obs_{}'.format(p)] = np.array([])

        for i in range(Nobs):
            yobs = np.random.normal(loc = 0, scale = R)
            xi = np.random.normal(loc = 0, scale = R, size = int(M[o]))
            xi = xi-xi.mean()
            vars()['obs_{}'.format(p)] = np.append(vars()['obs_{}'.format(p)], yobs)

            for j in range(Nout):
                eta = np.random.normal(size= (int(M[o]),1))
                if i == 0:
                    print('before {}'.format(vars()['z_{}'.format(p)].shape))
                    vars()['z_{}'.format(p)][:,-1] = (np.reshape(phi[p](z),(-1,1)) + eta)[:,0]
                    print('after {}'.format(vars()['z_{}'.format(p)].shape))
                else:
                    print('before {}'.format(vars()['z_{}'.format(p)].shape))
                    vars()['z_{}'.format(p)][:,-1] = (np.reshape(phi[p](vars()['z_{}'.format(p)][:,-1]),(-1,1)) + eta)[:,0]
                    print('after {}'.format(vars()['z_{}'.format(p)].shape))


            vars()['z_{}'.format(p)] = np.append(vars()['z_{}'.format(p)], np.reshape(phi[p](vars()['z_{}'.format(p)][:, -1]), (-1, 1)) + eta, axis=1)

            Pf = 1/(M[o]-1)*np.dot(vars()['z_{}'.format(p)][:,-1], (vars()['z_{}'.format(p)][:,-1]-np.mean(vars()['z_{}'.format(p)][:,-1])).T)
            # D = np.diag(np.ones(int(M[o]))) - 1/(M[o]-1)*(vars()['z_{}'.format(p)][:,-1]-np.mean(vars()['z_{}'.format(p)][:,-1]))/(Pf + R)*np.reshape((vars()['z_{}'.format(p)][:,-1] + xi - yobs), (-1,1))
            # vars()['z_{}'.format(p)][:, -1] = np.dot(vars()['z_{}'.format(p)][:, -1], D)
            # if yobs ~ N(0,4), also disturbing value xi ~ N(0,4)

            # linear approach
            K = Pf/(Pf + 4)
            vars()['z_{}'.format(p)][:, -1] =  vars()['z_{}'.format(p)][:, -1] - K*(vars()['z_{}'.format(p)][:,-1] + xi - yobs)
            # --------------
            vars()['rmse_{}'.format(p)] += (vars()['z_{}'.format(p)][:,-1].mean() - yobs)**2

        vars()['rmse_{}'.format(p)] = vars()['rmse_{}'.format(p)]/Nobs

    RMSEs = []
    for p in range(len(phi)):
        RMSEs.append(vars()['rmse_{}'.format(p)])

    return RMSEs



phi = [lambda z: -z, lambda z: z, lambda z: z**3]
M = np.linspace(10, 1000, 10)

valuesRMSE = []
for i in range(len(M)):
    valuesRMSE.append(RMSEcalc(M, phi, i))

for i in range(np.array(valuesRMSE).shape[1]):
    print(np.array(valuesRMSE)[:,i].mean())


plt.step(range(len(M)), np.array(valuesRMSE), where = 'mid')
plt.legend(('-z', 'z', 'z^3'))
plt.xticks(np.arange(10), M)
plt.xlabel('Ensemble size M')
plt.ylabel('RMSE')
plt.show()

