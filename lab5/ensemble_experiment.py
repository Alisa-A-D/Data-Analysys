import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

class GradientBoostingExperiment:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = {}
        self.baseline_score = 0 

    def _tune_hyperparameters(self):
        print(f"ЕТАП 1: Налаштування гіперпараметрів")
        
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(self.X_train, self.y_train)
        self.baseline_score = r2_score(self.y_val, tree.predict(self.X_val))
        start_time = time.time()
        
        param_grid = {
            'loss': ['squared_error', 'absolute_error', 'huber'],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.5, 0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        best_score = -float('inf')
        count = 0
        
        for loss in param_grid['loss']:
            for lr in param_grid['learning_rate']:
                for sub in param_grid['subsample']:
                    for max_f in param_grid['max_features']:
                        try:
                            model = GradientBoostingRegressor(
                                n_estimators=100, loss=loss, learning_rate=lr,
                                subsample=sub, max_features=max_f, random_state=42
                            )
                            model.fit(self.X_train, self.y_train)
                            score = r2_score(self.y_val, model.predict(self.X_val))

                            if score > best_score:
                                best_score = score
                                self.best_params = {
                                    'loss': loss, 'learning_rate': lr, 
                                    'subsample': sub, 'max_features': max_f
                                }
                        except ValueError:
                            continue
                        count += 1
        
        elapsed = time.time() - start_time

        print(f"Результати на перевірочній вибірці:")
        print(f"Ститистика пошуку:")
        print(f"  Перевірено комбінацій:  {count}")
        print(f"  Час виконання:          {elapsed:.2f} сек.\n")

        print(f"Ефективність (R2 Val):")
        print(f"  Baseline (Tree):        {self.baseline_score:.4f}")
        print(f"  Gradient Boosting:      {best_score:.4f}")
        print(f"  Приріст якості:        {best_score - self.baseline_score:+.4f}\n")
       
        print(f"Найкращі парметри:")
        for param, value in self.best_params.items():
            print(f"  {param:<15} : {value}")



    def _plot_comparison(self):
        estimators_range = range(10, 210, 10)
        gb_scores = []

        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(self.X_train, self.y_train)
        tree_test_score = r2_score(self.y_test, tree.predict(self.X_test))
        
        for n in estimators_range:
            model = GradientBoostingRegressor(n_estimators=n, **self.best_params, random_state=42)
            model.fit(self.X_train, self.y_train)
            gb_scores.append(r2_score(self.y_test, model.predict(self.X_test)))

        plt.figure(figsize=(10, 6))
        plt.plot(estimators_range, gb_scores, label='Gradient Boosting (Best Params)', marker='o')
        plt.axhline(y=tree_test_score, color='r', linestyle='--', label=f'Single Tree (R2={tree_test_score:.3f})')
        plt.xlabel("n_estimators")
        plt.ylabel("R2 Score (Test)")
        plt.title("Ефективність ансамблю: Gradient Boosting vs Decision Tree")
        plt.legend()
        plt.grid(True)
        plt.show()

    def _final_evaluation(self):
        print(f"\nЕТАП 2: фінальна оцінка та інтерпретація ")
        
        final_model = GradientBoostingRegressor(n_estimators=200, **self.best_params, random_state=42)
        final_model.fit(self.X_train, self.y_train)
        preds = final_model.predict(self.X_test)
        
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self.y_test, preds)
        
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(self.X_train, self.y_train)
        tree_r2 = r2_score(self.y_test, tree.predict(self.X_test))
        
        print(f"Результати на тестовій вибірці:")
        print(f"  Gradient Boosting R2: {r2:.5f}")
        print(f"  Decision Tree R2:     {tree_r2:.5f}")
        print(f"  Покращення:          {r2 - tree_r2:+.5f}")
        print(f"  RMSE помилка:         {rmse:.4f}")
        print(f"  MAPE помилка:         {mape:.2%}")
        
        print("\nВажливість ознак:")
        feature_names = ["Ознака 0", "Ознака 1", "Ознака 2", "Ознака 3"]
        importances = final_model.feature_importances_
        
        indices = importances.argsort()[::-1]
        for f in range(self.X_train.shape[1]):
            idx = indices[f]
            print(f"  {f+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    def run(self):
        print("\n === Пункт 3: побудова ансамблів === \n")
        self._tune_hyperparameters()
        self._plot_comparison()
        self._final_evaluation()

        return self.best_params