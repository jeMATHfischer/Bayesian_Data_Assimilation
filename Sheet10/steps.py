import matplotlib.pyplot as plt
import numpy as np
import random

a = np.arange(10)
np.random.shuffle(a)

plt.step(np.arange(10), a, where = 'mid')
plt.show()