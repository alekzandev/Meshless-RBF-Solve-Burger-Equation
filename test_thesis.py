import numpy as np
from Implement_Thesis import  *
from function_Fm import *

interpolation = interpolate(n=10)
mlist = np.vstack(tuple(interpolation.F()))
print()