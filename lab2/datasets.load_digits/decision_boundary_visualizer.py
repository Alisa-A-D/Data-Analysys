import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Клас 8: побудова границь рішень
class DecisionBoundaryVisualizer:

    def __init__(self, models, X, y):
        self.models = models
        self.X = X
        self.y = y
        self.classes = np.unique(y)

        self.pca = PCA(n_components=2)
        self.X_2d = self.pca.fit_transform(self.X)

    def plot_boundaries(self, resolution=300):
        x_min, x_max = self.X_2d[:, 0].min() - 1, self.X_2d[:, 0].max() + 1
        y_min, y_max = self.X_2d[:, 1].min() - 1, self.X_2d[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution))

        cmap = ListedColormap(plt.cm.tab10.colors[:len(self.classes)])
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            grid_points_64d = self.pca.inverse_transform(grid_points_2d)
        except Exception as e:
            print(f"Помилка при зворотному перетворенні PCA: {e}")
            return

        for name, model in self.models.items():
            Z = model.predict(grid_points_64d)
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            plt.scatter(
                self.X_2d[:, 0], 
                self.X_2d[:, 1],
                c=self.y, 
                cmap=cmap, 
                s=15, 
                edgecolor='k')
            
            plt.title(f"Границі рішень для моделі: {name}")
            plt.xlabel("PCA Компонент 1")
            plt.ylabel("PCA Компонент 2")
            plt.show()