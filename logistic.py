import numpy as np
import matplotlib.pyplot as plt


def log_gen(filename, plotting=False):
    # Data Setup
    base_data = np.genfromtxt(filename, delimiter=',')
    X = np.column_stack((base_data[:, 1], base_data[:, 2]))
    y = base_data[:, 0]

    # Class 0 data
    X_0 = X[y == 0]
    feature_1_0 = X_0[:, 0]
    feature_2_0 = X_0[:, 1]
    mean_arr_0 = np.array([[np.mean(feature_1_0),
                            np.mean(feature_2_0)]]).T

    # Class 1 data
    X_1 = X[y == 1]
    feature_1_1 = X_1[:, 0]
    feature_2_1 = X_1[:, 1]
    mean_arr_1 = np.array([[np.mean(feature_1_1),
                            np.mean(feature_2_1)]]).T

    # Cov Mat
    cov_mat = np.cov(np.stack((X[:, 0], X[:, 1]), axis=0))

    # Function to calculate statistic
    def ds_estimator(mean_0, mean_1, cov, x):
        f_h0_x = 0.5 * np.matmul(np.matmul((np.array([x]).T - mean_0).T, np.linalg.inv(cov)),
                                 (np.array([x]).T - mean_0))
        f_h1_x = (-0.5) * np.matmul(np.matmul((np.array([x]).T - mean_1).T, np.linalg.inv(cov)),
                                    (np.array([x]).T - mean_1))
        return f_h1_x + f_h0_x

    # Create decision statistic surface
    f1_min, f1_max = min(X[:, 0]), max(X[:, 0])
    f2_min, y2_max = min(X[:, 1]), max(X[:, 1])
    f1_mesh_points, f2_mesh_points = np.meshgrid(np.linspace(f1_min, f1_max, 200), np.linspace(f2_min, y2_max, 100))

    # Calculate decision statistics
    Z = np.c_[f1_mesh_points.ravel(), f2_mesh_points.ravel()]
    decision_stats = []
    for idx in range(0, Z.shape[0]):
        ds = ds_estimator(mean_arr_0, mean_arr_1, cov_mat, Z[idx])
        decision_stats.append(ds[0][0])

    ds_arr = np.array([decision_stats]).T
    ds_arr = ds_arr.reshape(f1_mesh_points.shape)

    # Plotting
    if plotting:
        plt.figure()
        plt.title("Logistic Discriminant Analysis λ(x)=0.5")
        plt.contourf(f1_mesh_points, f2_mesh_points, ds_arr)
        plt.colorbar()
        plt.contour(f1_mesh_points, f2_mesh_points, ds_arr, levels=[0.5], colors='black')
        plt.scatter(feature_1_0, feature_2_0, label='Class 0', edgecolors='white')
        plt.scatter(feature_1_1, feature_2_1, label='Class 1', edgecolors='white')
        plt.legend()
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    return ds_arr



