import numpy as np
import matplotlib.pyplot as plt
import math


def bayes_gen(filename, plotting=False):
    # Data Setup
    base_data = np.genfromtxt(filename, delimiter=',')
    X = np.column_stack((base_data[:, 1], base_data[:, 2]))
    y = base_data[:, 0]

    # Class 0 data
    X_0 = X[y == 0]
    feature_1_0 = X_0[:, 0]
    feature_2_0 = X_0[:, 1]
    cov_mat_0 = np.cov(np.stack((feature_1_0, feature_2_0), axis=0))
    mean_arr_0 = np.array([[np.mean(feature_1_0),
                            np.mean(feature_2_0)]]).T

    # Class 1 data
    X_1 = X[y == 1]
    feature_1_1 = X_1[:, 0]
    feature_2_1 = X_1[:, 1]
    cov_mat_1 = np.cov(np.stack((feature_1_1, feature_2_1), axis=0))
    mean_arr_1 = np.array([[np.mean(feature_1_1),
                            np.mean(feature_2_1)]]).T

    # Function to calculate statistic
    def likelihood_estimator(mean_0, mean_1, cov_0, cov_1, d, x):
        constant_0 = 1 / (((2 * math.pi) ** (d / 2)) * np.sqrt(np.linalg.det(cov_0)))
        constant_1 = 1 / (((2 * math.pi) ** (d / 2)) * np.sqrt(np.linalg.det(cov_1)))

        exp_0 = (-0.5) * np.matmul(np.matmul((np.array([x]).T - mean_0).T, np.linalg.inv(cov_0)),
                                   (np.array([x]).T - mean_0))

        exp_0 = math.exp(exp_0[0][0])

        exp_1 = (-0.5) * np.matmul(np.matmul((np.array([x]).T - mean_1).T, np.linalg.inv(cov_1)),
                                   (np.array([x]).T - mean_1))
        exp_1 = math.exp(exp_1[0][0])

        P_x_0 = constant_0 * exp_0
        P_x_1 = constant_1 * exp_1

        ln_likelihood = np.log(P_x_1 / P_x_0)

        return ln_likelihood

    # Create decision statistic surface
    f1_min, f1_max = min(X[:, 0]), max(X[:, 0])
    f2_min, y2_max = min(X[:, 1]), max(X[:, 1])
    f1_mesh_points, f2_mesh_points = np.meshgrid(np.linspace(f1_min, f1_max, 200), np.linspace(f2_min, y2_max, 100))

    # Calculate decision statistics
    Z = np.c_[f1_mesh_points.ravel(), f2_mesh_points.ravel()]
    ln_likelihoods = []
    for idx in range(0, Z.shape[0]):
        ln_l = likelihood_estimator(mean_arr_0, mean_arr_1, cov_mat_0, cov_mat_1, 2, Z[idx])
        ln_likelihoods.append(ln_l)

    ds_arr = np.array([ln_likelihoods]).T
    ds_arr = ds_arr.reshape(f1_mesh_points.shape)

    # Plotting
    if plotting:
        plt.figure()
        plt.title("Bayesian Classification ln(λ(x))=0")
        plt.contourf(f1_mesh_points, f2_mesh_points, ds_arr)
        plt.colorbar()
        plt.contour(f1_mesh_points, f2_mesh_points, ds_arr, levels=[0], colors='black')
        plt.scatter(feature_1_0, feature_2_0, label='Class 0', edgecolors='white')
        plt.scatter(feature_1_1, feature_2_1, label='Class 1', edgecolors='white')
        plt.legend()
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    return ds_arr





