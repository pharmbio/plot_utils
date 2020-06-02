import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_context("notebook")

# create data
x = np.array(['A', 'A', 'A', 'A', 'N', 'N', 'N', 'N'])
y = np.array(['Both', 'A', 'N', 'Null', 'Both', 'A', 'N', 'Null', ])
#z = np.array([9, 372, 391, 335, 1, 164, 2339, 1173])
z = np.array([5000, 372, 391, 335, 1, 2500, 2500, 2500])

z_scaled = 2500 * z / z.max()

# use the scatter function
plt.scatter(x, y, s=z_scaled)
plt.margins(.3)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Confidence: XXX")

for xi, yi, zi, z_si in zip(x, y, z, z_scaled):
    #print(xi,yi,zi,z_si)
    plt.annotate(zi, xy=(xi, yi), xytext=(np.sqrt(z_si)/2.+5, 0),
                 textcoords="offset points", ha="left", va="center")

sns.despine(offset=10, trim=True)

plt.show()
