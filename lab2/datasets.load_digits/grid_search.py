import pandas as pd
from sklearn.model_selection import GridSearchCV

#Клас 10: решітчастий пошук для підбору гіперпараметрів
class ModelGridSearcher:
    def __init__(self, model, param_grid, X_train, y_train, scoring=None, cv=5):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.cv = cv

    def run_search(self, verbose=True):
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=-1,
            verbose=1 if verbose else 0)

        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        print("\n===Пункт 11: результати решітчастого пошуку ===")
        print(f"Найкращі параметри: {best_params}")
        print(f"Найкращий середній показник CV: {best_score:.4f}")

        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df.sort_values(by='mean_test_score', ascending=False)
        print("\nТоп-5 комбінацій параметрів:")
        print(results_df[['params', 'mean_test_score']].head())

        return best_model, best_params, results_df
    