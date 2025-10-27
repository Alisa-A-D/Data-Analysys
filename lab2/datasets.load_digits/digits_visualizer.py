import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

#Клас 1: візуалізація початкових даних
class DigitsVisualizer:
    def __init__(self):
        self.data = load_digits()

    def show_samples(self, n_samples=10):
        images = self.data.images[:n_samples * 2]
        labels = self.data.target[:n_samples * 2]

        fig, axes = plt.subplots(2, n_samples, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'{labels[i]}')
            ax.axis('off')
        plt.suptitle('Приклади з набору digits ')
        plt.show()