import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman2

class FriedmanVisualizer:
    def __init__(self, n_samples=500, noise=0.0, random_state=42):
        self.X, self.y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)
        print(f"Дані згенеровано: {n_samples} зразків.")

    def plot_all_features(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Візуалізація даних Friedman-2', fontsize=16)

        feature_names = [
            'Ознака 0 (X0)', 
            'Ознака 1 (X1)', 
            'Ознака 2 (X2)', 
            'Ознака 3 (X3)']
        sc = None

        for i, ax in enumerate(axes.flatten()):
            sc = ax.scatter(self.X[:, i], self.y, c=self.y, cmap='viridis', 
                            edgecolors='k', s=30, alpha=0.7)
            
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel("Цільова змінна (y)")
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)
        cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
        fig.colorbar(sc, cax=cax, label="Значення y (колір)")

        plt.show()