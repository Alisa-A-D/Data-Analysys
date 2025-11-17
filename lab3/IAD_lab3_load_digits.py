import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler, label_binarize  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    classification_report)

GLOBAL_MAX_ITER = 500
GLOBAL_RANDOM_STATE = 42

#Пункт 1: візуалізація початкових даних
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

"""
Опис роботи класу:
Цей клас завантажує стандартний датасет рукописних цифр digits з sklearn.
Метод show_samples візуалізує задану кількість прикладів зображень 
у вигляді сітки за допомогою бібліотеки matplotlib.
"""
        

#Пункт 2: розбиття даних 
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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\n===Пункт 2: розбиття та масштабування даних ===")
        print(f"Розмір train: {X_train_scaled.shape}, test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
"""
Опис роботи класу:
Клас відповідає за підготовку даних перед навчанням.
Він завантажує повний датасет, розбиває його на тренувальну та тестову вибірки 
(використовуючи стратифікацію для збереження балансу класів).
Також виконує стандартизацію (StandardScaler).
"""

# Пункт 3.1 
class HiddenNeuronSearcher:
    def __init__(self, X_train, y_train, X_val=None, y_val=None, random_state=42, max_iter=500):
        if X_val is None or y_val is None:
            self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
        else:
            self.X_tr, self.y_tr = X_train, y_train
            self.X_val, self.y_val = X_val, y_val
        self.random_state = random_state
        self.max_iter = max_iter

    def search(self, start=1, step=1, max_neurons=200, target_acc=0.98, patience=3, verbose=True):
        best = None
        no_improve = 0
        best_acc = 0.0
        history = []

        for n in range(start, max_neurons + 1, step):
            m = MLPClassifier(hidden_layer_sizes=(n,), 
                  early_stopping=True,   
                  n_iter_no_change=10,     
                  validation_fraction=0.1, 
                  random_state=self.random_state, 
                  max_iter=self.max_iter)
            
            m.fit(self.X_tr, self.y_tr)
            y_val_pred = m.predict(self.X_val)
            acc = accuracy_score(self.y_val, y_val_pred)
            history.append((n, acc))

            if verbose:
                print(f"[HiddenNeuronSearcher] neurons={n} | val_acc={acc:.4f}")

            if acc > best_acc + 1e-6:
                best_acc = acc
                best = (n, m)
                no_improve = 0
            else:
                no_improve += 1

            if best_acc >= target_acc:
                if verbose:
                    print(f"[HiddenNeuronSearcher] Досягнуто цільової точності {target_acc:.3f} при {best[0]} нейронах.")
                break
            if no_improve >= patience:
                if verbose:
                    print(f"[HiddenNeuronSearcher] Зупинка: відсутнє покращення протягом {patience} кроків.")
                break

        if best:
            return {
                "best_n": best[0],
                "best_model": best[1],
                "best_acc": best_acc,
                "history": history}
        else:
            return {
                "best_n": None,
                "best_model": None,
                "best_acc": best_acc,
                "history": history}
        
"""
Опис роботи класу:  
Цей клас виконує пошук значення 'best_n' — мінімально достатньої кількості нейронів 
у прихованому шарі для досягнення заданої якості (наприклад, точності 98%).
Пошук відбувається динамічно: починаючи з малої кількості (start), клас ітеративно 
додає нейрони (з кроком step), навчає нову модель MLPClassifier і перевіряє її точність 
на валідаційній вибірці. Процес зупиняється, як тільки досягнуто цільової точності 
або якщо точність перестала зростати протягом заданої кількості кроків (patience).
"""

# Пункт 3.2: побудова моделей нейронної мережі
class NeuralModelsBuilder:
    def __init__(self, X_train, y_train, random_state=42, max_iter=500, alpha=0.0001):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.max_iter = max_iter
        self.alpha = alpha

    def _make_and_fit(self, hidden_tuple):
        name = self._name_from_hidden(hidden_tuple)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_tuple,
            activation='relu',
            solver='adam',
            alpha=self.alpha,
            early_stopping=False, 
            random_state=self.random_state,
            max_iter=self.max_iter)
        
        try:
            model.fit(self.X_train, self.y_train)
            print(f"Модель '{name}' навчена.")
        except Exception as e:
            print(f"Помилка навчання моделі '{name}': {e}")
            return name, None 
            
        return name, model

    @staticmethod
    def _name_from_hidden(hidden):
        if isinstance(hidden, (tuple, list)):
            if len(hidden) == 1:
                return f"mlp_1layer_{hidden[0]}"
            else:
                return "mlp_" + "_".join(map(str, hidden))
        else:
            return f"mlp_1layer_{int(hidden)}"

    def build_models(self, neurons_list=None):
        if neurons_list is None:
            neurons_list = [5, 10, 25, 50, 100]

        normalized = []
        for n in neurons_list:
            if isinstance(n, (tuple, list)):
                normalized.append(tuple(n))
            else:
                normalized.append((int(n),))

        models = {}

        for hidden in sorted(list(set(normalized))): 
            name, m = self._make_and_fit(hidden)
            if m: 
                models[name] = m

        return models

    def build_models_with_search(self, neurons_list=None, searcher=None, best_n=None, search_kwargs=None):
        
        print("\n=== Пункт 3: Побудова моделей ===")
        
        search_result = None

        if best_n is None and searcher is not None:
            kwargs = search_kwargs or {}
            search_result = searcher.search(**kwargs)
            best_n = search_result.get("best_n") or search_result.get("best_neurons")

            if best_n is not None:
                print(f"[NeuralModelsBuilder] Знайдено best_n = {best_n}.")
            else:
                print("[NeuralModelsBuilder] Searcher не знайшов рекомендованого best_n.")

        if neurons_list is None:
            neurons_list = [5, 10, 25, 50, 100]

        normalized_set = set()
        for n in neurons_list:
            if isinstance(n, (tuple, list)):
                normalized_set.add(tuple(n))
            else:
                normalized_set.add((int(n),))
 
        if best_n:
            if isinstance(best_n, (tuple, list)):
                candidate = tuple(best_n)
            else:
                candidate = (int(best_n),)
            
            if candidate not in normalized_set:
                 print(f"[NeuralModelsBuilder] Додаємо знайдену модель {candidate} до списку.")
                 normalized_set.add(candidate)
        
        build_list = sorted(list(normalized_set)) 

        models = self.build_models(neurons_list=build_list)
        return models, search_result
    
"""
Опис роботи класу:
Цей клас відповідає за масове створення та навчання моделей MLPClassifier.
Він приймає список бажаних архітектур (neurons_list) і, опціонально, використовує 
HiddenNeuronSearcher для знаходження та додавання оптимальної моделі до цього списку.
Повертає словник, де ключі — назви моделей, а значення — навчені об'єкти моделей.
"""
    
# Пункт 4: графічне представлення моделей
import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralModelVisualizer:

    def __init__(self, models):
        self.models = models

    def _plot_weight_images_separate(self, W, model_name, max_cols=8):
        n_features, n_hidden = W.shape

        img_shape = (8, 8) if n_features == 64 else None
        if img_shape is None: 
            return

        max_cols = min(max_cols, n_hidden)
        n_cols = max_cols
        n_rows = math.ceil(n_hidden / n_cols)

        fig_w = max(8, n_cols * 1.2)
        fig_h = max(4, n_rows * 1.3 + 1.6)
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        gs = fig.add_gridspec(n_rows, n_cols, top=0.88, bottom=0.18,
                              hspace=0.45, wspace=0.35)

        fig.suptitle(f"Ваги першого шару — {model_name}", fontsize=12)

        vmin, vmax = np.min(W), np.max(W)
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        im = None

        for idx in range(n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            ax = fig.add_subplot(gs[r, c])

            if idx < n_hidden:
                w = W[:, idx]
                img = w.reshape(img_shape)
                im = ax.imshow(img, cmap='seismic', vmin=vmin, vmax=vmax,
                               interpolation='nearest')
                ax.set_title(f"hn {idx}", fontsize=7, pad=2)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis('off')

        if im is not None:
            cax = fig.add_axes([0.15, 0.06, 0.7, 0.035])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8)

        fig.subplots_adjust(top=0.85, bottom=0.14, left=0.03, right=0.98)
        plt.show()

    def _plot_combined(self, W0, W_display, model_name, max_cols=8):
        n_features, n_hidden_display = W_display.shape
        img_shape = (8, 8) if n_features == 64 else None
        if img_shape is None:
            return

        max_cols = min(max_cols, n_hidden_display)
        n_cols = max_cols
        n_rows = math.ceil(n_hidden_display / n_cols)

        fig_width = max(10, n_cols * 1.5 + 4)
        fig_height = max(6, n_rows * 1.4 + 1)
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.suptitle(f"Аналіз ваг першого шару — {model_name}", fontsize=14, y=0.98)

        gs = fig.add_gridspec(n_rows + 1, n_cols, 
                              height_ratios=[1]*n_rows + [0.1], 
                              hspace=0.4, wspace=0.3)

        vmin, vmax = np.min(W_display), np.max(W_display)
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

        im = None
        for idx in range(n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            ax = fig.add_subplot(gs[r, c])

            if idx < n_hidden_display:
                w = W_display[:, idx]
                img = w.reshape(img_shape)
                im = ax.imshow(img, cmap='seismic', vmin=vmin, vmax=vmax,
                               interpolation='nearest')
                ax.set_title(f"hn {idx}", fontsize=7, pad=2)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

        if im is not None:
            cax = fig.add_subplot(gs[n_rows, :])
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.ax.tick_params(labelsize=8)

        plt.show()

    def plot_coefficients(self, max_cols=8, max_neurons_display=None):
        for name, model in self.models.items():
            if not hasattr(model, "coefs_") or model.coefs_ is None:
                print(f"[NeuralModelVisualizer]  У моделі '{name}' відсутні coefs_ ")
                continue
            try:
                W0 = model.coefs_[0]
            except Exception as e:
                print(f"[NeuralModelVisualizer] Не вдалося взяти coefs_[0] для '{name}': {e}")
                continue

            W_display = W0
            if max_neurons_display is not None and W0.shape[1] > max_neurons_display:
                W_display = W0[:, :max_neurons_display]

            if name == 'mlp_1layer_100':
                try:
                    self._plot_weight_images_separate(W_display, name, max_cols=max_cols)
                except Exception as e:
                    print(f"Помилка при побудові окремих графіків для '{name}': {e}")
            else:
                try:
                    self._plot_combined(W0, W_display, name, max_cols=max_cols)
                except Exception as e:
                    print(f"Помилка при побудові комбінованого графіка для '{name}': {e}")

    def plot_loss_curves(self):
        plt.figure(figsize=(10, 6))

        for name, model in self.models.items():
            if hasattr(model, "loss_curve_"):
                plt.plot(model.loss_curve_, label=name)

        plt.title("Криві втрат для моделей")
        plt.xlabel("Ітерації")
        plt.ylabel("Втрати")
        plt.legend(loc='best', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

"""
Опис роботи класу:
Цей клас призначений для візуальної інтерпретації нейронних мереж шляхом аналізу їхніх внутрішніх ваг та динаміки навчання. 
Метод plot_coefficients витягує вагові коефіцієнти першого прихованого шару та візуалізує їх у вигляді теплових карт, 
що дозволяє побачити, які саме візуальні патерни "вивчив" кожен нейрон, 
Метод plot_loss_curves збирає дані про процес тренування всіх моделей та будує зведений графік функції втрат.
"""
            
# Пункт 5: виконання прогнозів на основі побудованих моделей
class ModelPredictor:
    def __init__(self, models, X_train, X_test, y_train, y_test):
        self.models = models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def make_predictions(self, verbose=True):
        print("\n=== Пункт 5: виконання прогнозів на основі побудованих моделей ===")
        results = {}

        for name, model in self.models.items():
            try:
                if not hasattr(model, "classes_"):
                    model.fit(self.X_train, self.y_train)
            except Exception:
                pass

            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            y_proba_train = None
            y_proba_test = None

            if hasattr(model, "predict_proba"):
                try:
                    y_proba_train = model.predict_proba(self.X_train)
                    y_proba_test = model.predict_proba(self.X_test)
                except Exception:
                    y_proba_train = None
                    y_proba_test = None
            elif hasattr(model, "decision_function"):
                try:
                    df_train = model.decision_function(self.X_train)
                    df_test = model.decision_function(self.X_test)

                    def _to_proba(df):
                        df = np.atleast_2d(df)
                        exp = np.exp(df - np.max(df, axis=1, keepdims=True))
                        proba = exp / exp.sum(axis=1, keepdims=True)
                        return proba

                    y_proba_train = _to_proba(df_train)
                    y_proba_test = _to_proba(df_test)
                except Exception:
                    y_proba_train = None
                    y_proba_test = None

            results[name] = {
                "y_pred_train": y_pred_train,
                "y_pred_test": y_pred_test,
                "y_proba_train": y_proba_train,
                "y_proba_test": y_proba_test}

            if verbose:
                print(f"[{name}] Прогнози готові. Train preds: {len(y_pred_train)}, Test preds: {len(y_pred_test)}")

        return results 

"""
Опис роботи класу:
Цей клас автоматизує процес отримання прогнозів для набору моделей машинного навчання на навчальних та тестових даних. 
Метод make_predictions ітерується по всіх переданих моделях, генеруючи як класові мітки, 
так і ймовірнісні оцінки належності до класів. Клас містить логіку для уніфікації отримання ймовірностей: 
він намагається використати стандартний метод predict_proba, 
а за його відсутності — апроксимує ймовірності на основі decision_function, 
що дозволяє уніфікувати подальший розрахунок метрик для різних типів алгоритмів.
"""

# Пункт 6: перевірка перенавчання моделей
class OverfittingChecker:
    def __init__(self, models, predictions, X_train, X_test, y_train, y_test):
        self.models = models
        self.predictions = predictions
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_overfitting(self, 
                             gap_threshold=0.05, 
                             train_score_threshold=0.99):
  
        print("\n=== Пункт 6: оцінка перенавчання моделей ===")
        
        for name, model in self.models.items():
            if name not in self.predictions:
                print(f"\nПропуск моделі {name}: відсутні прогнози.")
                continue
                
            y_pred_train = self.predictions[name]["y_pred_train"]
            y_pred_test = self.predictions[name]["y_pred_test"]

            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            diff = acc_train - acc_test

            print(f"\nМодель: {name}")
            print(f"Train accuracy: {acc_train:.4f}")
            print(f"Test accuracy:  {acc_test:.4f}")
            print(f"Різниця (train - test): {diff:.4f}")

            if acc_train < train_score_threshold:
                print(f"Статус: Ознаки недонавчання!")
                print(f"  (Точність на train < {train_score_threshold*100}%)")
            
            elif diff > gap_threshold:
                print(f"Статус: Ознаки перенавчання!")
                print(f"  (Різниця > {gap_threshold*100}%)")
            
            else:
                print(f"Статус: Хороша здатність до узагальнення.")

"""
Опис роботи класу:
Клас для діагностики перенавчання та недонавчання.
Метод evaluate_overfitting аналізує точність на тренувальній та тестовій вибірках.
Він виявляє недонавчання, якщо точність на 'train' нижча за поріг, 
або перенавчання, якщо точність на 'train' висока, але різниця з 'test' перевищує встановлений поріг.
"""
                
# Пункт 7: розрахунок апостеріальних ймовірностей
class PosteriorProbabilitiesCalculator:
    def __init__(self, models, X_test, y_test):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

    def calculate_probabilities(self, n_samples=5):
        print(f"\n=== Пункт 7: розрахунок апостеріорних ймовірностей для {n_samples} прикладів ===")

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


"""
Опис роботи класу:
Цей клас призначений для глибшого аналізу якості класифікації шляхом перевірки "впевненості" моделі у своїх прогнозах. 
Метод calculate_probabilities відбирає задану кількість прикладів з тестової вибірки та обчислює для них апостеріорні ймовірності 
приналежності до кожного класу за допомогою методу predict_proba. Отримані результати зводяться у таблицю, 
яка відображає розподіл ймовірностей разом зі справжнім класом та фінальним прогнозом.
"""
                
# Пункт 8: побудова границь рішень
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

"""
Опис роботи класу:
Клас візуалізує границі рішень моделей у 2D просторі.
Оскільки дані багатовимірні (64 ознаки), використовується PCA для зменшення розмірності до 2.
Потім будується контурний графік, що показує, як модель класифікує різні області цього 2D простору.
"""

# Пункт 9: розрахунок метрик класифікації 
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

        metrics['report'] = classification_report(
            y_true,
            y_pred,
            target_names=[str(c) for c in self.classes],
            zero_division=0,
            digits=3)

        metrics['cm'] = confusion_matrix(y_true, y_pred)
    
        y_true_bin = label_binarize(y_true, classes=self.classes)

        metrics['fpr'], metrics['tpr'], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        metrics['roc_auc'] = auc(metrics['fpr'], metrics['tpr'])

        metrics['precision'], metrics['recall'], _ = precision_recall_curve(
            y_true_bin.ravel(),
            y_scores.ravel())
 
        metrics['pr_auc'] = auc(metrics['recall'], metrics['precision'])
    
        return metrics

    def evaluate_all(self):
        for name, model in self.models.items():

            if name not in self.predictions:
                continue
                
            y_pred_train = self.predictions[name]["y_pred_train"]
            y_pred_test = self.predictions[name]["y_pred_test"]
            y_scores_train = self.predictions[name].get("y_proba_train")
            y_scores_test = self.predictions[name].get("y_proba_test")

            if y_scores_train is None or y_scores_test is None:
                continue 

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
                     label=f'TRAIN PR-крива (AUC = {train_metrics["pr_auc"]:0.2f})')
            ax2.plot(test_metrics['recall'], test_metrics['precision'], color='green', lw=2,
                     label=f'TEST PR-крива (AUC = {test_metrics["pr_auc"]:0.2f})')
            
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('PR-крива (Train vs Test)')
            ax2.legend(loc="lower left")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plt.show()

"""
Опис роботи класу:
Цей клас виконує комплексну оцінку ефективності моделей класифікації, 
порівнюючи результати на навчальній та тестовій вибірках для виявлення перенавчання. 
Метод evaluate_all послідовно обробляє кожну модель, генеруючи детальні візуалізації: 
теплові карти матриць неточностей для аналізу помилок та графіки ROC і PR кривих для оцінки якості розділення класів. 
Розрахунок метрик покладено на внутрішній метод _get_metrics, який обчислює площі під кривими (AUC) 
та формує звіти класифікації, забезпечуючи всебічний аналіз поведінки алгоритмів.
"""
        
# Пункт 11: решітчастий пошук для підбору гіперпараметрів
class ModelGridSearcher:
    def __init__(self, model, param_grid, X_train, y_train, scoring=None, cv=5, n_jobs=-1,
                 use_randomized=False, n_iter=30, random_state=42, verbose=True):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.use_randomized = use_randomized
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def run_search(self):
        if self.use_randomized:
            searcher = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1 if self.verbose else 0)
        else:
            searcher = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=1 if self.verbose else 0)

        searcher.fit(self.X_train, self.y_train)

        best_params = searcher.best_params_
        best_score = getattr(searcher, "best_score_", None)
        best_model = searcher.best_estimator_

        print("\n=== Пункт 11: результати решітчастого пошуку ===")
        print(f"Найкращі параметри: {best_params}")
        if self.scoring and best_score is not None:
            print(f"Найкращий середній показник CV ({self.scoring}): {best_score:.4f}")
        elif best_score is not None:
            print(f"Найкращий середній показник CV: {best_score:.4f}")

        results_df = pd.DataFrame(searcher.cv_results_)
        if 'mean_test_score' in results_df.columns:
            results_df = results_df.sort_values(by='mean_test_score', ascending=False)
        else:
            results_df = results_df.sort_index()

        display_cols = [c for c in results_df.columns if c.startswith('param_')] + [c for c in ['mean_test_score', 'std_test_score'] if c in results_df.columns]
        display_cols = [c for c in display_cols if c in results_df.columns]
        if not display_cols:
            display_cols = results_df.columns.tolist()[:10] 

        print("\nТоп-5 комбінацій параметрів (за mean_test_score):")
        print(results_df[display_cols].head(5).to_string(index=False))

        return best_model, best_params, results_df
    
"""
Опис роботи класу:
Клас автоматизує підбір гіперпараметрів моделі (наприклад, кількість шарів, нейронів, альфа).
Він використовує GridSearchCV або RandomizedSearchCV 
з крос-валідацією для знаходження найкращої комбінації параметрів.
"""

# Пункт 13: навчання на підмножинах навчальних даних
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

"""
Опис роботи класу:
Цей клас досліджує ефективність моделі в залежності від обсягу доступних даних.
Метод evaluate_by_size проводить серію експериментів, 
навчаючи нові екземпляри моделі на випадкових підмножинах даних різного розміру та обчислюючи метрики якості.
Метод _plot_learning_curve візуалізує зібрану статистику, будуючи графік метрик (Train vs Test).
"""
        
if __name__ == "__main__":
    
    # 1. Візуалізація
    visualizer = DigitsVisualizer()
    visualizer.show_samples(n_samples=5)

    # 2. Розбиття даних
    splitter = DigitsDataSplitter(test_size=0.3)
    X_train, X_test, y_train, y_test = splitter.split()
    classes_list = np.unique(y_train)

    # 3. Побудова моделей
    builder = NeuralModelsBuilder(X_train, y_train, 
                                  random_state=GLOBAL_RANDOM_STATE, 
                                  max_iter=GLOBAL_MAX_ITER)
    searcher = HiddenNeuronSearcher(X_train, y_train, 
                                    random_state=GLOBAL_RANDOM_STATE, 
                                    max_iter=GLOBAL_MAX_ITER)
    models, search_res = builder.build_models_with_search(
        neurons_list=[5, 10, 25, 50, 100],
        searcher=searcher,
        search_kwargs={'start':5,
                        'step':5, 
                        'max_neurons':200, 
                        'target_acc':0.98, 
                        'patience':4, 
                        'verbose':False})


    # # 4. Графічне представлення моделей
    visualizer4 = NeuralModelVisualizer(models)
    visualizer4.plot_coefficients()
    visualizer4.plot_loss_curves()

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
        'hidden_layer_sizes': [(50,), (50,30), (100,)],
        'activation': ['relu', 'tanh'],
        'alpha': [1e-4, 1e-3],
        'solver': ['adam'],
        'learning_rate_init': [1e-3, 1e-2]}

    grid_searcher = ModelGridSearcher(
        model=MLPClassifier(max_iter=GLOBAL_MAX_ITER, 
                            random_state=GLOBAL_RANDOM_STATE),
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        scoring='accuracy',
        cv=5,
        use_randomized=False)

    best_model, best_params, grid_results = grid_searcher.run_search()

    # 13. Вплив розміру навчальної вибірки
    size_eval = TrainingSizeEvaluator(
        model=MLPClassifier(max_iter=GLOBAL_MAX_ITER, 
                            random_state=GLOBAL_RANDOM_STATE),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type='classification')

    learning_curve_df = size_eval.evaluate_by_size(train_sizes=np.linspace(0.1, 1.0, 6))
