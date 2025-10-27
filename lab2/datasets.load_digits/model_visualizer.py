import matplotlib.pyplot as plt
import numpy as np

#Клас 4: графічне представлення моделей
class LogisticModelVisualizer:
    def __init__(self, models):
        self.models = models

    def plot_coefficients(self):
        for name, model in self.models.items():
            coefs = model.coef_
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(f"Візуалізація коефіцієнтів моделі: {name}", fontsize=16)
            
            abs_max = np.abs(coefs).max()
            vmin, vmax = -abs_max, abs_max
            
            for i, ax in enumerate(axes.flat):
                if i < len(coefs):
                    image = coefs[i].reshape(8, 8)
                    im = ax.imshow(image, cmap='seismic', vmin=vmin, vmax=vmax)
                    
                    ax.set_title(f"Клас: {i}")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis('off') 

            fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05)
            plt.show()