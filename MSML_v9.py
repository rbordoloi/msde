import numpy as np
from scipy.spatial import cKDTree
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from scipy.sparse import csr_matrix
# import faiss  # For approximate nearest neighbor search
# from pynndescent import NNDescent   # <--- REMOVED
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from adbench.myutils_new import Utils
from adbench.run_new import RunPipeline
import pandas as pd
from numba import njit, prange


def count_points_within_radius(X, tree, epsilon):
    neighbors = tree.query_ball_tree(tree, epsilon)
    counts = np.array([len(pts) - 1 for pts in neighbors])
    return counts


def find_k_nearest_neighbors(X, tree, query_point, k):
    distances, indices = tree.query(query_point, k=k)
    return distances, indices


# --- UPDATED: remove NNDescent ---
def max_min_distances_kdtree(X):
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=X.shape[0])   # full neighbor matrix
    all_distances = dists[:, 1:].flatten()
    max_distance = np.max(all_distances)
    min_distance = np.min(all_distances)
    return max_distance, min_distance


def binary_search_condition(low, high, condition, tol=1e-4, max_iter=50):
    result = None
    for _ in range(max_iter):
        mid = (low + high) / 2
        if condition(mid):
            result = mid
            high = mid
        else:
            low = mid
        if abs(high - low) < tol:
            break
    return result


def condition_formulation(point_in_radius_counts, nbd_sample_count_threshold, satisfiability_proportion):
    if_satisfied = (np.where(point_in_radius_counts > nbd_sample_count_threshold)[0]) * 1
    return np.sum(if_satisfied) >= satisfiability_proportion


def t_distribution_kernel(X, nu=1.0):
    G = X @ X.T
    diag = np.diag(G)
    D = diag[:, None] + diag[None, :] - 2 * G
    K = 1 / (1 + D) ** nu
    return K + 1e-7


def get_empirical_weights(
    X,
    nbd_sample_count_threshold=5,
    max_iters_weight_count=4,
    satisfiability_proportion=0.3,
    n_neighbors=15,
    metric='euclidean',
    random_state=42,
    batch_size=1000
):
    def umap_graph_similarity(X_batch):
        knn_indices, knn_dists, _ = nearest_neighbors(
            X_batch,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
            low_memory=True,
            use_pynndescent=True
        )
        G, _, _ = fuzzy_simplicial_set(
            X_batch,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            apply_set_operations=True,
            verbose=False
        )
        return G.toarray() if isinstance(G, csr_matrix) else G

    def compute_weights_from_similarity(sim, X_batch):
        tree = cKDTree(sim)

        # Use KDTree for min/max distances
        max_dist, min_dist = max_min_distances_kdtree(sim)

        threshold = (
            max(1, len(X_batch) - 1)
            if nbd_sample_count_threshold >= len(X_batch)
            else nbd_sample_count_threshold
        )
        effective_required = int(satisfiability_proportion * len(X_batch))

        eps = binary_search_condition(
            min_dist, max_dist,
            lambda mid: condition_formulation(
                count_points_within_radius(sim, tree, mid),
                threshold,
                effective_required
            )
        )

        if eps is None:
            relaxed_thresh = max(1, threshold // 2)
            relaxed_prop = effective_required // 2
            eps = binary_search_condition(
                min_dist, max_dist,
                lambda mid: condition_formulation(
                    count_points_within_radius(sim, tree, mid),
                    relaxed_thresh,
                    relaxed_prop
                )
            )

        if eps is None:
            eps = max_dist

        delta = (eps - 1e-6) / max_iters_weight_count
        all_counts = []

        for _ in range(max_iters_weight_count):
            counts = count_points_within_radius(sim, tree, eps)
            all_counts.append(counts)
            eps -= delta

        return np.mean(all_counts, axis=0)

    if len(X) <= 3 * n_neighbors:
        sim = umap_graph_similarity(X)
        return compute_weights_from_similarity(sim, X)

    effective_batch_size = min(batch_size, max(n_neighbors * 3, 100))
    total_batches = (len(X) + effective_batch_size - 1) // effective_batch_size

    weights_all = []

    for batch_idx in range(total_batches):
        start = batch_idx * effective_batch_size
        end = min(len(X), start + effective_batch_size)
        X_batch = X[start:end]
        sim = umap_graph_similarity(X_batch)
        weights_batch = compute_weights_from_similarity(sim, X_batch)
        weights_all.append(weights_batch)

    return np.concatenate(weights_all)


@njit(fastmath=True, parallel=True)
def shift_data(X, indices, weights, learning_rate):

    indicesShape = indices.shape
    data_nebs = X[indices.ravel()].reshape(indicesShape[0], indicesShape[1], X.shape[1])
    weights_nebs = weights[indices.ravel()].reshape(indicesShape[0], indicesShape[1])
    weights_nebs = weights_nebs / (weights_nebs.sum(axis=1).reshape(-1, 1) + 1e-6)

    new_d = (weights_nebs.reshape(-1, weights_nebs.shape[1], 1) * data_nebs).sum(axis=1)

    change = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        change[i] = np.linalg.norm(X[i] - new_d[i])

    unit_vec = (new_d - X) / (change.reshape(-1, 1) + 1e-6)
    revised_d = X + learning_rate * change.reshape(-1, 1) * unit_vec

    return revised_d, change


# === UPDATED get_shift_fast: remove NNDescent ===
def get_shift_fast(X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold, return_weights=False):

    weights = get_empirical_weights(
        X,
        nbd_sample_count_threshold=nbd_sample_count_threshold,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        batch_size=1000
    )

    n_samples = X.shape[0]
    shifted_dataset = X.copy()
    total_distance = np.zeros(n_samples)

    for iter_count in range(max_iters_shift):

        # rebuild KDTree at each iteration
        tree = cKDTree(shifted_dataset)

        # KDTree kNN search
        dists, indices = tree.query(shifted_dataset, k)

        revised_d, change = shift_data(shifted_dataset, indices, weights, learning_rate)

        total_distance += change
        shifted_dataset = revised_d

        if change.mean() < shift_threshold:
            break

    if return_weights:
        return shifted_dataset, weights, total_distance
    else:
        return shifted_dataset, total_distance


def mean_shift_manifold_learning(X, k=30, nbd_sample_count_threshold=30, learning_rate=.3, max_iters_shift=10, shift_threshold=0.0001, return_weights=False):

    if not return_weights:
        data_shifted, total_distance = get_shift_fast(
            X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold
        )
        return data_shifted, total_distance
    else:
        data_shifted, weights, total_distance = get_shift_fast(
            X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold, return_weights=True
        )
        return data_shifted, weights, total_distance


class MSML:
    def __init__(self, seed: int, model_name: str = 'MSML', k=100, nbd_sample_count_threshold=70, learning_rate=.1, max_iters_shift=6, shift_threshold=0.003, anomalyThreshold=0.22, scaler=StandardScaler()):
        self.k = k
        self.nbd_sample_count_threshold = nbd_sample_count_threshold
        self.learning_rate = learning_rate
        self.max_iters_shift = max_iters_shift
        self.shift_threshold = shift_threshold
        self.anomalyThreshold = anomalyThreshold
        self.scaler = scaler
        self.seed = seed
        self.utils = Utils()
        self.model_name = model_name

    def fit(self, X_train, y_train=None):
        return self

    def predict_score(self, X):
        data_shifted, total_distance = mean_shift_manifold_learning(
            X,
            self.k,
            self.nbd_sample_count_threshold,
            self.learning_rate,
            self.max_iters_shift,
            self.shift_threshold
        )

        total_distance = self.scaler.fit_transform(total_distance.reshape(-1, 1))
        total_distance = expit(total_distance)
        return total_distance.squeeze()


if __name__ == "__main__":

    utils = Utils()
    utils.download_datasets()

    pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode='cluster', noise_type='irrelevant_features')
    results = pipeline.run(clf=MSML)
    pd.DataFrame(results).to_csv('adbench/result/MSML21.csv', index=False)

    results2 = pipeline.run()
    pd.DataFrame(results2).to_csv('adbench/result/benchmarks21.csv', index=False)
