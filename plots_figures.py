import matplotlib.pyplot as plt
import seaborn as sns

from halton_points import HaltonPoints


def N(n, d): return (2**n + 1)**d


def fill_distance(n): return 2**(-n)


n, d = 7, 2

M = HaltonPoints(2, N(n, d)).haltonPoints()

print (fill_distance(n))

x = M[:, 0]
y = M[:, 1]

ax = sns.scatterplot(x, y, color='#51F5CA', alpha=0.8,
                     linewidth=0.6, edgecolor='#4543FA')
ax.set(xlabel='$x$', ylabel='$y$')
plt.show()
