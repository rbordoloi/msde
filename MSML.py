import numpy as np
import torch
from sklearn.manifold import MDS
from scipy.spatial import KDTree, cKDTree
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
import umap
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import faiss  # For approximate nearest neighbor search
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from adbench.myutils import Utils
from adbench.run import RunPipeline
import pandas as pd



def count_points_within_radius(X, tree, epsilon):
    neighbors = tree.query_ball_tree(tree, epsilon)
    counts = np.array([len(pts) - 1 for pts in neighbors])
    return counts

def find_k_nearest_neighbors(X, tree, query_point, k):
    distances, indices = tree.query(query_point, k=k)
    return distances, indices

def max_min_distances_kdtree(X):
    tree = cKDTree(X)
    distances, _ = tree.query(X, k=len(X), p=2)
    all_distances = distances[:, 1:].flatten()
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
        max_dist, min_dist = max_min_distances_kdtree(sim)
        if nbd_sample_count_threshold >= len(X_batch):
            # print(f"[Warning] Threshold {nbd_sample_count_threshold} exceeds batch size {len(X_batch)}. Adjusting...")
            threshold = max(1, len(X_batch) - 1)
        else:
            threshold = nbd_sample_count_threshold

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
            # print(f"[Warning] Binary search failed (batch size {len(X_batch)}). Trying relaxed condition...")
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
            # print("[Warning] Relaxed binary search also failed. Using max_dist as fallback.")
            eps = max_dist

        delta = (eps - 1e-6) / max_iters_weight_count
        all_counts = []

        for i in range(max_iters_weight_count):
            #print(f'  Iteration {i+1}/{max_iters_weight_count} | radius = {eps:.4f}')
            counts = count_points_within_radius(sim, tree, eps)
            all_counts.append(counts)
            eps -= delta

        return np.mean(all_counts, axis=0)

    if len(X) <= 3 * n_neighbors:
        # print('Using full-data UMAP similarity (small dataset).')
        sim = umap_graph_similarity(X)
        return compute_weights_from_similarity(sim, X)

    # print('Using batched UMAP similarity (large dataset).')
    effective_batch_size = min(batch_size, max(n_neighbors * 3, 100))
    total_batches = (len(X) + effective_batch_size - 1) // effective_batch_size
    weights_all = []

    # for batch_idx in tqdm(range(total_batches), desc="Calculating empirical weights"):
    for batch_idx in range(total_batches):
        start = batch_idx * effective_batch_size
        end = min(len(X), start + effective_batch_size)
        #print(f'Processing batch {batch_idx + 1}/{total_batches}')
        X_batch = X[start:end]
        sim = umap_graph_similarity(X_batch)
        weights_batch = compute_weights_from_similarity(sim, X_batch)
        weights_all.append(weights_batch)

    return np.concatenate(weights_all)

def shift_data_torch(X, indices, weights, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.from_numpy(X).float().to(device)
    indices_t = torch.from_numpy(indices).long().to(device)
    weights_t = torch.from_numpy(weights).float().to(device)

    batch_size, k = indices_t.shape
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)

    data_nebs = X_t[indices_t]
    weights_nebs = weights_t[indices_t]
    weights_nebs = weights_nebs / (weights_nebs.sum(dim=1, keepdim=True) + 1e-6)

    new_d = torch.sum(weights_nebs.unsqueeze(2) * data_nebs, dim=1)
    change = torch.norm(X_t - new_d, dim=1)
    unit_vec = (new_d - X_t) / (change.unsqueeze(1) + 1e-6)
    revised_d = X_t + learning_rate * change.unsqueeze(1) * unit_vec

    return revised_d.cpu().numpy(), change.cpu().numpy()

def get_shift_fast(X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold, return_weights=False):
    # print('Generating tree and calculating sample weights...')
    weights = get_empirical_weights(
        X,
        nbd_sample_count_threshold=nbd_sample_count_threshold,
        max_iters_weight_count=4,
        satisfiability_proportion=0.3,
        batch_size=1000
    )

    # print('Shifting data...')
    n_samples = len(X)
    shifted_dataset = X.copy()
    total_distance = np.zeros(n_samples)  # <-- Track total distance travelled per point

    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.astype(np.float32))

    for iter_count in range(max_iters_shift):
        # print(f'Iteration {iter_count + 1}/{max_iters_shift}')
        changes = []

        # for start in tqdm(range(0, n_samples, X.shape[0])):
        for start in range(0, n_samples, X.shape[0]):
            end = min(start + X.shape[0], n_samples)
            d = shifted_dataset[start:end]
            _, indices = index.search(d.astype(np.float32), k)
            revised_d, change = shift_data_torch(shifted_dataset, indices, weights, learning_rate)

            # accumulate total distance travelled per point
            total_distance[start:end] += change

            shifted_dataset[start:end] = revised_d
            changes.extend(change.tolist())

        avg_change = np.mean(changes)
        # print(f'Average change: {avg_change:.6f}')
        if avg_change < shift_threshold:
            # print('Converged!')
            break

    #return (shifted_dataset, weights) if return_weights else shifted_dataset
    if return_weights:
        return shifted_dataset, weights, total_distance
    else:
        return shifted_dataset, total_distance

def mean_shift_manifold_learning(X, k=30, nbd_sample_count_threshold=30, learning_rate=.3, max_iters_shift=10, shift_threshold=0.0001, return_weights=False):
    # if return_weights==False:
    #     data_shifted = get_shift_fast(X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold)
    # else:
    #     data_shifted, weights = get_shift_fast(X, k, nbd_sample_count_threshold, learning_rate, max_iters_shift, shift_threshold, return_weights=True)

    # return data_shifted if not return_weights else (data_shifted, weights)

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
    def __init__(self, seed: int, model_name: str = 'MSML', k=12, nbd_sample_count_threshold=70, learning_rate=.1, max_iters_shift=6, shift_threshold=0.003, anomalyThreshold=0.22, scaler=StandardScaler()):
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
        # dataMSML = mean_shift_manifold_learning(X, self.k, self.nbd_sample_count_threshold, self.learning_rate, self.max_iters_shift, self.shift_threshold)
        # dataMSMLShifts = np.linalg.norm(X - dataMSML, axis=1).squeeze()

        # dataMSMLShifts = self.scaler.fit_transform(dataMSMLShifts.reshape(-1, 1))

        # Scale and apply sigmoid
        total_distance = self.scaler.fit_transform(total_distance.reshape(-1, 1))
        total_distance = expit(total_distance)
        return total_distance.squeeze()
    
        # return dataMSMLShifts.squeeze()
    
    # def predict(self, X):
    #     """Return anomaly scores (same as predict_score)."""
    #     return self.predict_score(X)

    # def __call__(self, X):
    #     """Allow model(X) syntax (used by ADBench pipeline)."""
    #     return self.predict_score(X)
    

if __name__ == "__main__":

    utils = Utils()
    utils.download_datasets()

    pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode='dependency', noise_type='irrelevant_features')
    results = pipeline.run(clf=MSML)
    pd.DataFrame(results).to_csv('adbench/result/MSML2.csv', index=False)
    results2 = pipeline.run()
    pd.DataFrame(results2).to_csv('adbench/result/benchmarks2.csv', index=False)