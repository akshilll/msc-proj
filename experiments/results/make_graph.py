import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open("betweenness_results.pickle", "rb")
f2 = open("closeness_results.pickle", "rb")
f3 = open("eigenvector_results.pickle", "rb")
f4 = open("katz_results.pickle", "rb")
f5 = open("load_results.pickle", "rb")

out = pickle.load(f)
out2 = pickle.load(f2)
out3 = pickle.load(f3)
out4 = pickle.load(f4)
out5 = pickle.load(f5)

f.close()
f2.close()
f3.close()
f4.close()
f5.close()

plt.plot(np.array(out).mean(0))
plt.plot(np.array(out2).mean(0))
plt.plot(np.array(out3).mean(0))
plt.plot(np.array(out4).mean(0))
plt.plot(np.array(out5).mean(0))
plt.show()