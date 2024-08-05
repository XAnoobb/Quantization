import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(1000)

plt.hist(data, bins=50)

plt.title("histgram")
plt.xlabel("value")
plt.ylabel("freq")
# plt.savefig("histgram.png", bbox_inches="tight")
plt.show()
