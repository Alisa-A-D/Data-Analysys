import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, r2_score

#Клас 11: навчання на підмножинах навчальних даних
class TrainingSizeEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test, task_type='classification'):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type

    def evaluate_by_size(self, train_sizes=np.linspace(0.1, 1.0, 6)):
        results = []
        print("\n===Пункт 13: навчання на підмножинах ===")
        for size in train_sizes:
            X_subset, y_subset = resample(
                self.X_train, self.y_train,
                replace=False,
                n_samples=int(len(self.X_train) * size),
                random_state=42)

            model_clone = type(self.model)(**self.model.get_params())
            model_clone.fit(X_subset, y_subset)

            y_pred_train = model_clone.predict(self.X_train)
            y_pred_test = model_clone.predict(self.X_test)

            if self.task_type == 'classification':
                score_train = accuracy_score(self.y_train, y_pred_train)
                score_test = accuracy_score(self.y_test, y_pred_test)
                metric_name = "Accuracy"
            else:
                score_train = r2_score(self.y_train, y_pred_train)
                score_test = r2_score(self.y_test, y_pred_test)
                metric_name = "R2"

            results.append({
                "train_size_fraction": round(size, 2),
                f"{metric_name}_train": score_train,
                f"{metric_name}_test": score_test})

            print(f"Навчено на {int(size*100)}% даних | "
                  f"Train {metric_name}: {score_train:.3f} | Test {metric_name}: {score_test:.3f}")
        
        df_results = pd.DataFrame(results)
        self._plot_learning_curve(df_results, metric_name)
        return df_results

    def _plot_learning_curve(self, df, metric_name):
        plt.figure(figsize=(7, 5))
        plt.plot(df["train_size_fraction"], df[f"{metric_name}_train"], marker='o', label="Train")
        plt.plot(df["train_size_fraction"], df[f"{metric_name}_test"], marker='s', label="Test")
        plt.title(f"Вплив розміру навчальної вибірки на {metric_name}")
        plt.xlabel("Частка навчальної множини")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.show()