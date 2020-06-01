import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_color_codes()

# create data
x = np.array(['A', 'A', 'A', 'A', 'N', 'N', 'N', 'N'])
y = np.array(['Both', 'A', 'N', 'Null', 'Both', 'A', 'N', 'Null', ])
z = np.array([9, 372, 391, 335, 1, 164, 2339, 1173])

# use the scatter function
plt.scatter(x, y, s=z*1)
plt.margins(.3)

for xi, yi, zi in zip(x, y, z):
    plt.annotate(zi, xy=(xi, yi), xytext=(np.sqrt(zi)/2.+5, 0),
                 textcoords="offset points", ha="left", va="center")

plt.show()
