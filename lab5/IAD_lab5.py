from sklearn.datasets import make_friedman2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import time
import numpy as np
import matplotlib.pyplot as plt

class FriedmanDataSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        
    def split(self, test_size=0.2, val_size=0.2, random_state=42):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)        
        if (1 - test_size) == 0:
            raise ValueError("Size of set can't be 1.0")            
        val_ratio = val_size / (1 - test_size)        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state)
        return True

    def show_split_sizes(self):
        print("\n === Step 2: data splitt === \n" )
        print("Splitting results:")
        print(f"Whole set:      {len(self.X)}")
        print(f"Learning set:   {len(self.X_train)}")
        print(f"Validation set: {len(self.X_val)}")
        print(f"Testing set:    {len(self.X_test)}")
class FriedmanVisualizer:
    def __init__(self, n_samples=500, noise=0.0, random_state=42):
        self.X, self.y = make_friedman2(n_samples=n_samples, noise=noise, random_state=random_state)
        print(f"Data generated: {n_samples} samples.")
    def plot_all_features(self):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle('Visualization of Friedman-2', fontsize=16)
        feature_names = [
            'Feature 0 (X0)', 
            'Feature 1 (X1)', 
            'Feature 2 (X2)', 
            'Feature 3 (X3)']
        sc = None
        for i, ax in enumerate(axes.flatten()):
            sc = ax.scatter(self.X[:, i], self.y, c=self.y, cmap='viridis', 
                            edgecolors='k', s=30, alpha=0.7)            
            ax.set_xlabel(feature_names[i])
            ax.set_ylabel("Target variable (y)")
            ax.grid(True, linestyle='--', alpha=0.6)
        plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)
        cax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
        fig.colorbar(sc, cax=cax, label="Values y (color)")
        plt.show()
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
        self.final_ensemble = None
    def _tune_hyperparameters(self):
        print(f"STEP 1: Fitting hyperparameters")        
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

        print(f"Results for testing set:")
        print(f"Search statistics:")
        print(f"  Checked combinations: {count}")
        print(f"  Execution time:       {elapsed:.2f} s.\n")

        print(f"Efficiency (R2 Val):")
        print(f"  Baseline (Tree):        {self.baseline_score:.4f}")
        print(f"  Gradient Boosting:      {best_score:.4f}")
        print(f"  Quality increment:     {best_score - self.baseline_score:+.4f}\n")
       
        print(f"Best parameters:")
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
        plt.title("Ansamble efficiency: Gradient Boosting vs Decision Tree")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def _final_evaluation(self):
        print(f"\nSTEP 2: final evaluation and interpretation ")
        
        self.final_ensemble = GradientBoostingRegressor(n_estimators=200, **self.best_params, random_state=42)
        self.final_ensemble.fit(self.X_train, self.y_train)
        preds = self.final_ensemble.predict(self.X_test)
        
        r2 = r2_score(self.y_test, preds)
        mse = mean_squared_error(self.y_test, preds)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(self.y_test, preds)
        
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(self.X_train, self.y_train)
        tree_r2 = r2_score(self.y_test, tree.predict(self.X_test))
        
        print(f"Test set results:")
        print(f"  Gradient Boosting R2: {r2:.5f}")
        print(f"  Decision Tree R2:     {tree_r2:.5f}")
        print(f"  Improvement:         {r2 - tree_r2:+.5f}")
        print(f"  RMSE:         {rmse:.4f}")
        print(f"  MAPE:         {mape:.2%}")
        
        print("\nFeatures importance:")
        feature_names = ["Feature 0", "Feature 1", "Feature 2", "Feature 3"]
        importances = self.final_ensemble.feature_importances_
        
        indices = importances.argsort()[::-1]
        for f in range(self.X_train.shape[1]):
            idx = indices[f]
            print(f"  {f+1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
    def plot_predictions(self):
        ensemble = self.final_ensemble
        base_estimator = ensemble.estimators_[0][0]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Gradient Boosting Regression predictions', fontsize=16)
        feature_names = ['X0','X1','X2','X3']
        for i, ax in enumerate(axes.flatten()):
            X_range = np.linspace(self.X_test[:, i].min(), self.X_test[:, i].max(), 200).reshape(-1, 1)
            X_mean = self.X_test.mean(axis=0)
            X_plot = np.tile(X_mean, (200, 1))
            X_plot[:, i] = X_range.flatten()
            
            y_ensemble_pred = ensemble.predict(X_plot)
            y_base_pred = base_estimator.predict(X_plot)
            ax.scatter(self.X_test[:, i], self.y_test, s=30, alpha=0.5, color='gray', label='Actual objects $y_{true}$')
            ax.plot(X_range, y_ensemble_pred, color='red', linewidth=3, label='Ensemble prediction (Gradient Boosting)')
            ax.plot(X_range, y_base_pred, color='navy', linestyle='--', linewidth=2, label='Base model prediction (First decision tree)')
            ax.set_xlabel(f'Feature {feature_names[i]}', fontsize=12)
            ax.set_ylabel('Target variable $y$', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels)
        plt.show()   
        
    def MSE_Bias_Varience(self):
        model = self.final_ensemble
        base_estimator = model.estimators_[0][0]
        # Settings
        n_repeat = 50 
        n_train = 300  
        n_test = 100
        noise = 0.1  
        estimators = [("DecisonTreeRegressor", base_estimator),("GradientBoostingRegressor", model),]
        def f_friedman2(X):
            X0 = X[:, 0]
            X1 = X[:, 1]
            X2 = X[:, 2]
            X3 = X[:, 3]
            return np.sqrt(X0**2 + (X1 * X2 - 1 / (X1 * X3))**2)
        # --- Data generation ---
        _X_train = []
        _y_train = []
        for i in range(n_repeat):
            X, y = make_friedman2(n_samples=n_train, noise=noise, random_state=i + n_repeat)
            _X_train.append(X)
            _y_train.append(y)    
        _X_test, _ = make_friedman2(n_samples=n_test, noise=noise, random_state=42)
        _y_test = np.zeros((n_test, n_repeat))
        for i in range(n_repeat):
            _y_test[:, i] = f_friedman2(_X_test) + np.random.normal(0.0, noise, n_test)    
        
        print("--- Bias-variance-noise (make_friedman2) ---")
        for n, (name, estimator) in enumerate(estimators):
            y_predict = np.zeros((n_test, n_repeat))
            for i in range(n_repeat):
                estimator.fit(_X_train[i], _y_train[i])
                y_predict[:, i] = estimator.predict(_X_test)
            # MSE
            y_error = np.zeros(n_test)
            for i in range(n_repeat):
                for j in range(n_repeat):
                    y_error += (_y_test[:, j] - y_predict[:, i]) ** 2
            y_error /= n_repeat **2 
            # Noise
            y_noise = np.var(_y_test, axis=1)
            # Bias^2
            y_bias = (f_friedman2(_X_test) - np.mean(y_predict, axis=1)) ** 2
            # Variance
            y_var = np.var(y_predict, axis=1)
            print("{0}: {1:.4f} (MSE) = {2:.4f} (Bias^2) + {3:.4f} (Variance) + {4:.4f} (Noise)".format(
                name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise))
            )
            
    def speed_evaluation(self):
        estimator = self.final_ensemble
        start_time = time.time()
        estimator.fit(self.X_train, self.y_train)
        GBR_duration = time.time() - start_time
        print(f"Learning duration GBR (N={estimator.n_estimators}) on X/Y_train: {GBR_duration:.4f} s")
        whole_duration, min_d, max_d = 0, [0, 1], [0, 0]
        for i in range(estimator.n_estimators):
            base_estimator = estimator.estimators_[i][0]
            start = time.time()
            base_estimator.fit(self.X_train, self.y_train)
            single_duration = time.time()-start
            whole_duration += single_duration
            if single_duration > max_d[1]:
                max_d = [i, single_duration]
            if single_duration < min_d[1]:
                min_d = [i, single_duration]
        print(f"Whole duration of consecutive learning of {estimator.n_estimators} base models on X/Y_train: {whole_duration:.4f} s")
        print(f"Average learning duration of base_model on X/Y_train: {whole_duration/estimator.n_estimators:.4f} s")
        print(f"Max learning duration of base_model {max_d[0]} in the set: {max_d[1]:.4f} s")
        print(f"Min learning duration of base_model {min_d[0]} in the set: {min_d[1]:.4f} s")
        
    def run(self):
        print("\n === Step 3: ensembles assembling === \n")
        self._tune_hyperparameters()
        self._plot_comparison()
        self._final_evaluation()
        print("\n === Step 4: plot predictions === \n")
        self.plot_predictions()
        print("\n === Step 5: calculating MSE, Bias, Varience === \n")
        self.MSE_Bias_Varience()
        print("\n === Step 6: speed evaluation === \n")
        self.speed_evaluation()

if __name__ == "__main__":
    # Step 1: generating and visualizing data
    handler = FriedmanVisualizer()
    handler.plot_all_features()
    # Step 2: splitting data
    splitter = FriedmanDataSplitter(handler.X, handler.y)
    splitter.split(test_size=0.2, val_size=0.2)
    splitter.show_split_sizes()
    # Step 3-6
    experiment = GradientBoostingExperiment(
        splitter.X_train, splitter.y_train,
        splitter.X_val, splitter.y_val,
        splitter.X_test, splitter.y_test)
    experiment.run()
