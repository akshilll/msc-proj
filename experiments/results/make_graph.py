import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open("betweenness_results.pickle", "rb")
f2 = open("closeness_results.pickle", "rb")
f3 = open("degree_results.pickle", "rb")
f4 = open("eigenvector_results.pickle", "rb")
f5 = open("katz_results.pickle", "rb")
f6 = open("load_results.pickle", "rb")
f7 = open("primitive_results.pickle", "rb")

out = pickle.load(f)
out2 = pickle.load(f2)
out3 = pickle.load(f3)
out4 = pickle.load(f4)
out5 = pickle.load(f5)
out6 = pickle.load(f6)
out7 = pickle.load(f7)

f.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()

# plt.plot(np.array(out).mean(0), label="betweenness")
# plt.plot(np.array(out2).mean(0), label="closeness")
# plt.plot(np.array(out3).mean(0), label="degree")
# plt.plot(np.array(out4).mean(0), label="eigenvector")
# plt.plot(np.array(out5).mean(0), label="katz")
plt.plot(np.array(out6).mean(0), label="load")
plt.plot(np.array(out7).mean(0), label="primitive")
plt.grid()
plt.legend()
plt.show()