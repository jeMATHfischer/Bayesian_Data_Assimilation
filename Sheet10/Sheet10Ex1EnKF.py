import numpy as np
import matplotlib.pyplot as plt

M = np.linspace(10,1000,100)
phi = [lambda z: -z, lambda z: z, lambda z: z**3]

o = 50 # Index for M

P = 2
R = 1/4

Nobs = 1000 # Nout = 1?
Nout = 1
# H = 1
z = np.random.normal(loc = 0, scale = P, size = (int(M[o]),1))

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
                vars()['z_{}'.format(p)] = np.append(vars()['z_{}'.format(p)],
                                                     np.reshape(phi[p](z), (-1, 1)) + eta,
                                                     axis=1)
                print('after {}'.format(vars()['z_{}'.format(p)].shape))
            else:
                print('before {}'.format(vars()['z_{}'.format(p)].shape))
                vars()['z_{}'.format(p)] = np.append(vars()['z_{}'.format(p)], np.reshape(phi[p](vars()['z_{}'.format(p)][:,-1]),(-1,1)) + eta, axis = 1)
                print('after {}'.format(vars()['z_{}'.format(p)].shape))

        Pf = 1/(M[o]-1)*np.dot(vars()['z_{}'.format(p)][:,-1], (vars()['z_{}'.format(p)][:,-1]-np.mean(vars()['z_{}'.format(p)][:,-1])).T)
        # D = np.diag(np.ones(int(M[o]))) - 1/(M[o]-1)*(vars()['z_{}'.format(p)][:,-1]-np.mean(vars()['z_{}'.format(p)][:,-1]))/(Pf + R)*(vars()['z_{}'.format(p)][:,-1] + xi - yobs)
        # vars()['z_{}'.format(p)][:, -1] = np.dot(D.T, vars()['z_{}'.format(p)][:, -1])
        # if yobs ~ N(0,4), also disturbing value xi ~ N(0,4)

        # linear approach
        K = Pf/(Pf + R)
        vars()['z_{}'.format(p)][:, -1] =  vars()['z_{}'.format(p)][:, -1] - K*(vars()['z_{}'.format(p)][:,-1] + xi - yobs)
        # # --------------
        vars()['rmse_{}'.format(p)] += (vars()['z_{}'.format(p)][:,-1].mean()-yobs)**2

    vars()['rmse_{}'.format(p)] = vars()['rmse_{}'.format(p)]/Nobs


RMSEs = []
for p in range(len(phi)):
    RMSEs.append(vars()['rmse_{}'.format(p)])

fig, ax = plt.subplots(1,3)
ax[0].plot(np.append(np.arange(Nobs), Nobs), z_0[20].T)
ax[1].plot(np.append(np.arange(Nobs), Nobs), z_1[20].T)
ax[2].plot(np.append(np.arange(Nobs), Nobs), z_2[20].T)
plt.show()

print(RMSEs)

plt.plot(np.arange(len(phi)), RMSEs)
plt.show()
