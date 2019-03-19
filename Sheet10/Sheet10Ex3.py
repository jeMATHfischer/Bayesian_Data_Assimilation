import numpy as np
import matplotlib.pyplot as plt


Nx = [1,10,100,200]

def pdf(y,N):
    r = y*np.sqrt(N)
    # return (r)**(N-1)*np.exp(-r**2/2)
    return (r)**(N-1)*np.exp(-r**2/2)

t = np.linspace(0,2,1000)

fig, (ax1,ax10,ax100,ax200) = plt.subplots(1,4)
for item in Nx:
    vars()['ax{}'.format(item)].plot(t, pdf(t,item))
    vars()['ax{}'.format(item)].set_title('Nx = {}'.format(item))
plt.show()