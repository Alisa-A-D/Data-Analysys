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
                dist = self.dist_matrix[i, j]
                if dist < min_val:
                    min_val = dist
                    c1, c2 = i, j
        return c1, c2, min_val
        
    def merge_clusters(self, c1, c2):
        keys = list(self.clusters.keys())
        temp = self.clusters[keys[c1]] + self.clusters[keys[c2]]
        new_index = max(self.clusters.keys()) + 1
        self.clusters[new_index] = temp
        del self.clusters[keys[c1]]
        del self.clusters[keys[c2]]
        
    def single_cluster_distance(self, a, b):
        temp = (a + b)/2
        return temp
        
    def clusters_distance(self, c1, c2):
        n = len(self.clusters) - 1
        new_matrix = self.dist_matrix
        new_matrix = np.delete(new_matrix, [c1, c2], axis=0)
        new_matrix = np.delete(new_matrix, [c1, c2], axis=1)
        temp = np.zeros((n, n))        
        temp[:n-1, :n-1]=new_matrix
        for i in range(c1):
            temp[i,n-1]=self.single_cluster_distance(self.dist_matrix[c1, i], self.dist_matrix[c2,i])
            temp[n-1,i]=self.single_cluster_distance(self.dist_matrix[c1, i], self.dist_matrix[c2,i])
        for i in range (c1, c2):
            step = c2 - c1 
            if step > 1:
                step -= 1  
                temp[i,n-1]=self.single_cluster_distance(self.dist_matrix[c1, i+1], self.dist_matrix[c2,i+1])
                temp[n-1,i]=self.single_cluster_distance(self.dist_matrix[c1, i+1], self.dist_matrix[c2,i+1])
            elif step == 1:
                break
        for i in range (n, c2, -1):
            if i == c2:
                break
            else:   
                temp[n-1,i-2]=self.single_cluster_distance(self.dist_matrix[c1, i], self.dist_matrix[c2,i])
                temp[i-2,n-1]=self.single_cluster_distance(self.dist_matrix[c1, i], self.dist_matrix[c2,i])
        return temp
    
    def run(self):
        step = 1
        while len(self.clusters) > 1:
            keys = list(self.clusters.keys())
            print("Distance matrix:")
            print(self.dist_matrix)
            c1, c2, dist = self.find_closest_clusters()
            print(f"\n=== Step {step} ===")
            print(f"Closest clusters: {keys[c1]} and {keys[c2]} (distance = {dist:.3f})")
            self.dist_matrix = self.clusters_distance(c1, c2)
            self.merge_clusters(c1, c2)
            print("Current clusters:")
            for k, v in self.clusters.items():
                print(f"  {k}: {v}")
            step += 1
