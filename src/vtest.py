# import numpy
import numpy as np

# import vdbscan
from vdbscan import VariantDBSCAN

# Cretae simple dataset
x = np.array([3, 2, 8, 1])
y = np.array([3, 2, 8, 1])

# Set epsilon and minpoints values
eps = np.array([np.sqrt(2)+0.01, np.sqrt(2)-0.01])
mp = np.array([2,2])

# Cluster data
res = VariantDBSCAN(eps,mp,x,y)

# Print results
print(res)

