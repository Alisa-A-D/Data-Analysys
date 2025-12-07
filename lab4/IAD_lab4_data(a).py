from sklearn.datasets import make_circles
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

#Step 1: Visualization
def data_visualizer(X, Y):
    plt.figure(figsize=(10,8))
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
    plt.title("Step 1: Data set 'circles'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
#Step 2: Building model
def model_builder(X, params):
    model = OPTICS(min_samples=params[0], xi=params[1], min_cluster_size=params[2], metric=params[3]).fit(X)
    if model != None:
        print(f"Model build successfully!")
    n_noise_ = list(model.labels_).count(-1)
    print(f"OPTICS: noise points amount: {n_noise_}.")
    return model
#Step 3: Execute clustering
def execute_clustering(X, model):
    results = model.fit_predict(X)
    if len(results)==0:
        print("Failed execution.")
    else:
        print(f"Successful execution.")
    return results
#Step 4: Clustering visualization
def clustering_visualization(X, model, results):
    plt.figure(figsize=(10,8))
    plt.scatter(X[:,0],X[:,1], c=results, edgecolor='k')
    plt.title('Clusters')
    plt.show()
    plt.figure(figsize=(12,5))    
    plt.plot(np.arange(len(X)), model.reachability_[model.ordering_], marker='.', color='navy')
    plt.title('OPTICS Reachability Plot')
    plt.xlabel('Order of points')
    plt.ylabel('Reachability distance')
    plt.show()    
#Step 5: Speed evaluation
def speed_evaluation(data_sizes, model):
    results = {}
    for i in data_sizes:
        temp, _ = make_circles(n_samples=i, noise=0.1, factor=0.1, random_state=42)
        start = time.time()
        model.fit(temp)
        end = time.time()
        results[i]=end-start
    print(results)
    plt.figure(figsize=(8,6))
    plt.plot(data_sizes, results.values(), marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("Execution time/data amount dependency")
    plt.xlabel("Data amount")
    plt.ylabel("Execution time")
    plt.show()
#Step 6: Alternative models
def alternative_models(X):
    options = {'min_samples':[3, 8, 15], 'xi':[0.01, 0.1, 0.2],
    'min_cluster_size':[0.01, 0.2, 0.35], 'metric': ['cityblock','cosine','chebyshev']}
    alternative = {'min_samples':{}, 'xi':{}, 'min_cluster_size':{}, 'metric':{}}
    for parameter in options.keys():
       print("--------------")
       for i in options[parameter]:
          print(f"{parameter}: {i}")
          values=[(i if parameter == 'min_samples' else 10), (i if parameter == 'xi' else 0.05),
            (i if parameter == 'min_cluster_size' else 0.05), (i if parameter == 'metric' else 'euclidean')]
          alternative[parameter][i] = model_builder(X, values)
    return alternative
#Step 7: Quality metrics
def quality_metrics(models, X, Y):
    results = {}
    for param in models:
        for option in models[param].keys():
            labels = models[param][option].labels_
            n_clusters = len(np.unique(labels[labels != -1]))
            ari = adjusted_rand_score(Y, labels)
            ami = adjusted_mutual_info_score(Y, labels)
            silhouette = silhouette_score(X, models[param][option].fit_predict(X))
            results[f"{param}: {option}"]={'Estimated clusters': n_clusters,'ARI': ari,'AMI': ami, 'Silhouette score': silhouette}
    df = pd.DataFrame.from_dict(results,orient='index')
    print(df)  
#Step 8: Analyze
def check_stability(X, model, labels1):
    ari = np.empty(5)
    for i in range(len(ari)):
        idx = np.random.permutation(len(X))
        labels2 = model.fit_predict(X[idx])
        labels2_unshuffled = labels2[np.argsort(idx)]
        ari[i] = adjusted_rand_score(labels1, labels2_unshuffled)
    print(f"ARI for every shuffle iteration:\n{ari}\nAverage ARI: {np.mean(ari)}")
  
if __name__ == "__main__":
    X, Y = make_circles(n_samples=10000, noise=0.1, factor=0.1, random_state=42)
    data_visualizer(X, Y)
    print(f"-----Step 2: Building model-----")
    clustering = model_builder(X,[10,0.05,0.05,'euclidean'])
    print(f"-----Step 3: Executing model-----")
    results = execute_clustering(X,clustering)
    clustering_visualization(X, clustering, results)
    print(f"-----Step 5: Algorithm's speed evaluation-----")
    speed_evaluation([1000,10000,50000,100000], clustering)
    print(f"-----Step 6: Building alternative models-----")
    alternative = alternative_models(X)
    print(f"-----Step 7: Quality assessment-----")
    quality_metrics(alternative, X, Y)
    print(f"-----Step 8: Stability check-----")
    check_stability(X, clustering, results)
