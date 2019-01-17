#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = [0.5, 0.3, 0.1, 0.05, 0.01]



# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:, 0:3]

# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]


# Initialize w.


# Error values over all iterations.
error = []
DATA_FIG = 1

# Set up the slope-intercept figure
# SI_FIG = 2
# plt.figure(SI_FIG)
# plt.rcParams.update({'font.size': 15})
# plt.title('Separator in slope-intercept space')
# plt.xlabel('slope')
# plt.ylabel('intercept')
# plt.axis([-5, 5, -10, 0])

w = np.array([0.1, 0, 0])
print(data[2, 3])
#     error.append(e_all[-1])
# e_all =[]
# eta = [0.5]
# w = np.array([0.1, 0, 0])
# for etta in eta:
#         for iter in range(0, max_iter):
#                 for i in range (1, 200):
#                         d = data[i-1:i, 0:3]
#                         target = data[i, 3]
#
#                         y = sps.expit(np.dot(d, w))
#
#                         e = -np.mean(np.multiply(target, np.log(y)) + np.multiply((1 - target), np.log(1 - y)))
#                         e_all.append(e)
#
#                         grad_e = np.mean(np.multiply((y - target), d.T), axis=1)
#
#                         # Update w, *subtracting* a step in the error derivative since we're minimizing
#                         w_old = w
#                         w = w - etta * grad_e
#
#                         plt.figure(SI_FIG)
#                         a2.plot_mb(w, w_old)
#                         print(e)
#
#
#                 print 'epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T)
#                 if iter > 0:
#                         if np.absolute(e - e_all[iter - 1]) < tol:
#                                 break
#                 # Stop iterating if error doesn't change more than tol.
#

# for iter in range(0, max_iter):
#         randArray = np.random.permutation(X.shape[0])
#         for rand in randArray:

eta = [0.5, 0.3, 0.1, 0.05, 0.01]
#randArray = np.random.permutation(X.shape[0])
#randArray = np.random.permutation(X.shape[0])
plt.figure()
for etta in eta:
    w = np.array([0.1, 0, 0])
    SGD_errors = []
    np.random.seed(757)
    randArray = np.random.permutation(X.shape[0])

    for iter in range(0, max_iter):
            e_all = []
            for rand in randArray:
                        example = data[rand, 0:3]
                        example_target = data[rand, 3]

                        y = sps.expit(np.dot(example, w))

                        if(y != 1.0 ):
                            e = -np.mean(np.multiply(example_target, np.log(y)) + np.multiply((1-example_target), np.log(1-y)))
                        e_all.append(e)

                        grad_e = np.multiply((y - example_target), example.T)

                        w_old = w
                        w = w - etta * grad_e

            if iter > 0:
                    if np.absolute(np.mean(e_all) - SGD_errors[iter - 1]) < tol:
                            break
            SGD_errors.append(np.mean(e_all))

            # plt.plot(SGD_errors)
    plt.plot(SGD_errors,label= etta)

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression SGD for different eta')
plt.xlabel('Epoch')
plt.legend()
plt.show()









