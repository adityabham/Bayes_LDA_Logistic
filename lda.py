import numpy as np
import matplotlib.pyplot as plt


def lda_gen(filename, plotting=False):
    # Data Setup
    base_data = np.genfromtxt(filename, delimiter=',')
    X = np.column_stack((base_data[:, 1], base_data[:, 2]))
    y = base_data[:, 0]
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    num_classes = len(np.unique(y))
    num_features = np.size(X, axis=1)
    class_indexes = []
    for i in range(num_classes):
        class_indexes.append(np.argwhere(y == i))

    # Training
    m0 = np.mean(X[class_indexes[0]], axis=0)
    diff_0 = np.subtract(X[class_indexes[0]].reshape(-1, num_features), m0)
    SW_0 = diff_0.T.dot(diff_0)

    m1 = np.mean(X[class_indexes[1]], axis=0)
    diff_1 = np.subtract(X[class_indexes[1]].reshape(-1, num_features), m1)
    SW_1 = diff_1.T.dot(diff_1)

    SW = SW_0 + SW_1
    W = np.dot(np.linalg.pinv(SW), np.subtract(m1, m0).reshape(-1, 1))

    # Create decision statistic surface
    f1_min, f1_max = min(X[:, 0]), max(X[:, 0])
    f2_min, y2_max = min(X[:, 1]), max(X[:, 1])
    f1_mesh_points, f2_mesh_points = np.meshgrid(np.linspace(f1_min, f1_max, 200), np.linspace(f2_min, y2_max, 100))

    # Calculate decision statistics
    Z = np.c_[f1_mesh_points.ravel(), f2_mesh_points.ravel()]
    decision_stats = []
    for j in range(len(Z)):
        ds = np.dot(W.T, Z[j])
        decision_stats.append(ds[0])

    ds_arr = np.array(decision_stats).reshape(f1_mesh_points.shape)

    if plotting:
        plt.figure()
        plt.title("Linear Discriminant Analysis λ(x)=0")
        plt.contourf(f1_mesh_points, f2_mesh_points, ds_arr)
        plt.colorbar()
        plt.contour(f1_mesh_points, f2_mesh_points, ds_arr, levels=[0], colors='black')
        plt.scatter(X_0[:, 0], X_0[:, 1], label='Class 0', edgecolors='white')
        plt.scatter(X_1[:, 0], X_1[:, 1], label='Class 1', edgecolors='white')
        plt.legend()
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    return ds_arr






