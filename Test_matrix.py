from expressions import *
from halton_points import HaltonPoints

Mi =Mi = HaltonPoints(2, 30).haltonPoints()

Mb = np.array([
    [0., 0.],
    [0.5, 1.]
])

print(Mi)


#x = np.array([0.113, 0.846])

#operators = operators(Mi, Mb, 2, 0, x)

#print(operators.K1())

