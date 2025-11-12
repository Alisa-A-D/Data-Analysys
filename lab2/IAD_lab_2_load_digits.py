import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample



warnings.simplefilter(action='ignore', category=FutureWarning)

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
        
        
#Клас 2: розбиття даних
class DigitsDataSplitter:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        digits = load_digits()
        X = digits.data
        y = digits.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        print("\n===Пункт 2: розбиття даних на навчальний та валідаційний набори ===")
        print(f"Розмір train: {X_train.shape}, test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
#Клас 3: побудова моделей логістичної регресії
class LogisticModelsBuilder:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
     
    def build_models(self):
        models = {}

        models["simple_regularized"] = LogisticRegression(max_iter=1000)
        models["simple_no_reg"] = LogisticRegression(C=1e6, penalty="l2", max_iter=1000)

        models["multinomial_regularized"] = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000)
        models["multinomial_no_reg"] = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=1e6, penalty="l2", max_iter=1000)

        print("\n===Пункт 3: побуова даних та навчання моделі ===")
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            print(f"Модель '{name}' навчено.")

        return models
    
    
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
            
 #Клас 5: виконання прогнозів на основі побудованих моделей           
class ModelPredictor:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def make_predictions(self):
        print("\n=== Пункт 5: виконання прогнозів на основі побудованих моделей ===")
        results = {}
       
        for name, model in self.models.items():
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            results[name] = {
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test}

            print(f"Прогнози виконано для моделі: {name}")

        return results
    
 #Клас 6: перевірка перенавчання моделей
class OverfittingChecker:
    def __init__(self, models, predictions, X_train, X_test, y_train, y_test):
        self.models = models
        self.predictions = predictions
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_overfitting(self):
        print("\n=== Пункт 6: оцінка перенавчання моделей ===")
        for name, model in self.models.items():
            y_pred_train = self.predictions[name]["y_pred_train"]
            y_pred_test = self.predictions[name]["y_pred_test"]

            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            diff = acc_train - acc_test

            print(f"\nМодель: {name}")
            print(f"Train accuracy: {acc_train:.4f}")
            print(f"Test accuracy:  {acc_test:.4f}")
            print(f"Різниця (train - test): {diff:.4f}")

            if diff > 0.05:
                print("Ознаки перенавчання!")
            else:
                print("Перенавчання не виявлено.")
                
                
#Клас 7: розрахунок апостеріальних ймовірностей
class PosteriorProbabilitiesCalculator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

    def calculate_probabilities(self, n_samples=5):
        print(f"\n=== Пункт 7: розрахунок апостеріорних ймовірностей для перших {n_samples} прикладів ===")

        for name, model in self.models.items():
            probs = model.predict_proba(self.X_test[:n_samples])

            df_probs = pd.DataFrame(
                np.round(probs, 3),
                columns=[f"Клас {i}" for i in range(probs.shape[1])])
            
            df_probs["Справжній клас"] = self.y_test[:n_samples].astype(int)
            df_probs["Прогноз"] = model.predict(self.X_test[:n_samples])

            print(f"\nМодель: {name}")
            with pd.option_context('display.width', 1000, 
                                 'display.precision', 3,
                                 'display.colheader_justify', 'center'):
                print(df_probs.to_string(index=False))
                
                
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


# Клас 9: розрахунок метрик класифікації без текстового виводу
class ClassificationEvaluator:
    def __init__(self, models, predictions, X_train, y_train, X_test, y_test, classes):
        self.models = models
        self.predictions = predictions
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test
        self.classes = classes
        self.n_classes = len(classes)

    def _get_metrics(self, y_true, y_pred, y_scores):
        metrics = {}

        metrics['report'] = classification_report(y_true, y_pred, target_names=[str(c) for c in self.classes])

        metrics['cm'] = confusion_matrix(y_true, y_pred)

        y_true_bin = label_binarize(y_true, classes=self.classes)

        metrics['fpr'], metrics['tpr'], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        metrics['roc_auc'] = auc(metrics['fpr'], metrics['tpr'])

        metrics['precision'], metrics['recall'], _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
        
        return metrics

    def evaluate_all(self):
        for name, model in self.models.items():
            y_pred_train = self.predictions[name]["y_pred_train"]
            y_pred_test = self.predictions[name]["y_pred_test"]

            y_scores_train = model.predict_proba(self.X_train)
            y_scores_test = model.predict_proba(self.X_test)

            train_metrics = self._get_metrics(self.y_train, y_pred_train, y_scores_train)
            test_metrics = self._get_metrics(self.y_test, y_pred_test, y_scores_test)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Матриці неточностей для {name}', fontsize=16)

            sns.heatmap(train_metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.classes, yticklabels=self.classes, ax=ax1)
            ax1.set_title('TRAIN дані')
            ax1.set_xlabel('Прогноз')
            ax1.set_ylabel('Справжнє значення')

            sns.heatmap(test_metrics['cm'], annot=True, fmt='d', cmap='Greens', 
                        xticklabels=self.classes, yticklabels=self.classes, ax=ax2)
            ax2.set_title('TEST дані')
            ax2.set_xlabel('Прогноз')
            ax2.set_ylabel('Справжнє значення')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Криві якості (Micro-average) для {name}', fontsize=16)

            ax1.plot(train_metrics['fpr'], train_metrics['tpr'], color='blue', lw=2, 
                     label=f'TRAIN ROC (AUC = {train_metrics["roc_auc"]:0.2f})')
            ax1.plot(test_metrics['fpr'], test_metrics['tpr'], color='green', lw=2, 
                     label=f'TEST ROC (AUC = {test_metrics["roc_auc"]:0.2f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC-крива (Train vs Test)')
            ax1.legend(loc="lower right")

            ax2.plot(train_metrics['recall'], train_metrics['precision'], color='blue', lw=2,
                     label='TRAIN PR-крива')
            ax2.plot(test_metrics['recall'], test_metrics['precision'], color='green', lw=2,
                     label='TEST PR-крива')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('PR-крива (Train vs Test)')
            ax2.legend(loc="lower left")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plt.show()
        
        
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
        
        
if __name__ == "__main__":
    
    # 1. Візуалізація
    visualizer = DigitsVisualizer()
    visualizer.show_samples(n_samples=5)

    # 2. Розбиття даних
    splitter = DigitsDataSplitter(test_size=0.3)
    X_train, X_test, y_train, y_test = splitter.split()
    classes_list = np.unique(y_train)

    # 3. Побудова моделей
    builder = LogisticModelsBuilder(X_train, y_train)
    models = builder.build_models()

    # 4. Графічне представлення моделей
    visualizer4 = LogisticModelVisualizer(models)
    visualizer4.plot_coefficients()

    # 5. Прогнози моделей
    predictor = ModelPredictor(models, X_train, X_test, y_train, y_test)
    predictions = predictor.make_predictions()

    # 6. Перевірка перенавчання
    overfit_checker = OverfittingChecker(models, predictions, X_train, X_test, y_train, y_test)
    overfit_checker.evaluate_overfitting()
    
    # 7. Апостеріорні ймовірності
    posterior_calc = PosteriorProbabilitiesCalculator(models, X_test, y_test)
    posterior_calc.calculate_probabilities(n_samples=5)

    # 8. Графічне представлення границь рішень
    boundary_viz = DecisionBoundaryVisualizer(models, X_train, y_train)
    boundary_viz.plot_boundaries()
    
    # 9. Критерії якості класифікації 
    metrics_calc = ClassificationEvaluator(
        models=models,
        predictions=predictions,
        X_train=X_train,
        y_train=y_train,
        y_test=y_test,
        X_test=X_test,     
        classes=classes_list )
    metrics_calc.evaluate_all()
    
    # 11. Підбір гіперпараметрів 
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'] }

    grid_searcher = ModelGridSearcher(
        model=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring='accuracy',
        cv=5)

    best_model, best_params, grid_results = grid_searcher.run_search()
     
    # 13. Вплив розміру навчальної вибірки 
    size_eval = TrainingSizeEvaluator(
        model=LogisticRegression(max_iter=1000),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type='classification')

    learning_curve_df = size_eval.evaluate_by_size(train_sizes=np.linspace(0.1, 1.0, 6))