import numpy as np

class Clustering:
    def __init__(self, values):
        self.values = values
        self.n = values.shape[0]
        self.clusters = {i: [i] for i in range(self.n)}
        self.dist_matrix = self._calculate_initial_distances()

    def _cluster_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _calculate_initial_distances(self):
        n = self.n
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = self._cluster_distance(self.values[i], self.values[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix


    def find_closest_clusters(self):
        keys = list(self.clusters.keys())
        min_val = np.inf
        c1, c2 = -1, -1

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                dist = self.dist_matrix[ki, kj]
                if dist < min_val:
                    min_val = dist
                    c1, c2 = ki, kj

        return c1, c2, min_val

    def merge_clusters(self, c1, c2):
        new_index = max(self.clusters.keys()) + 1
        self.clusters[new_index] = self.clusters[c1] + self.clusters[c2]
        del self.clusters[c1]
        del self.clusters[c2]

    def run(self):
        step = 1
        while len(self.clusters) > 1:
            c1, c2, dist = self.find_closest_clusters()
            print(f"\n=== Step {step} ===")
            print(f"Closest clusters: {c1} and {c2} (distance = {dist:.3f})")

            self.merge_clusters(c1, c2)

            print("Current clusters:")
            for k, v in self.clusters.items():
                print(f"  {k}: {v}")

            print("Distance matrix:")
            print(self.dist_matrix)

            step += 1