# Import Libraries
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
# Need to set backend before importing pyplot
# This script writes graphs to a file
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Use the folling line if variantdbscan has been installed using pip
# from vdbscan import VariantDBSCAN
# Otherwise, comment out the above line and uncomment the following line
from variantdbscan import VariantDBSCAN


# Read in test data set
data = pd.read_csv("test_dataset.csv", header=None,names=['x','y'])

# Create two varients
eps = np.array([1, 0.5])
mp = np.array([10, 12])

# Run variant dbscan
res = VariantDBSCAN(eps,mp, data.x.as_matrix(), data.y.as_matrix(), 70, verbose=False)


# Loop over the results and create plots
for i in range(len(eps)):

    # Uncomment the following lines to output
    # information about the results
    
    # print('For epsilon:', eps[i], 'and Min Points:', mp[i])
    # print('\tNumber of noise points:', np.sum(res[:,i]==0))
    # print('\tNumber of clusters:    ', np.max(res[:,i]))

    # Loop over each detected cluster and plot it
    for j in range(0,np.max(res[:,i])+1):
        plt.plot(data.loc[res[:,i] == j, 'x'],
                 data.loc[res[:,i] == j, 'y'], '.')

    # Set a title, and save the plot 
    plt.title('Epsilon:' + str(eps[i]) + ', Min Points: ' + str(mp[i]))
    plt.savefig('test_results_' + str(i) + '.pdf');
    plt.close();

    # Save the result in the original data frame
    data.loc[:,'cluster_' + str(i)] = res[:,i]

# Loop over and print results
# Made to match output of C example code
for i in range(len(eps)):
    print("\n\nVariant: %i, Epsilon: %f, minpts: %d" % (i,eps[i],mp[i]), end='');
    for label, row in data.iterrows():
        print("\n%d, %f, %f, %d" % (label, row['x'], row['y'], row['cluster_' + str(i)]), end='')
print()
    
# Save results to a file
data.to_csv('test_output.csv',sep=',', index=False, header=False,float_format='%.4f')

