import time
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score



class DataGenerator:
    def __init__(self):
        self.centers = [[0.0, 0.0], [3.5, 3.5]]
        self.cluster_std = [1.5, 0.5]
        self.X = None
        self.y_true = None
        self._generate_data()

    def _generate_data(self):
        self.X, self.y_true = make_blobs(
            n_samples=[1000, 700],
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=0,
            shuffle=False
        )
        print("Початкові дані згенеровано.")

    def get_data(self):
        return self.X, self.y_true

    def generate_custom_size(self, n_total_samples):
        n_centers = len(self.centers)
        
        samples_per_blob = [n_total_samples // n_centers] * n_centers
        samples_per_blob[0] += n_total_samples % n_centers 

        X_new, y_new = make_blobs(
            n_samples=samples_per_blob,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=42, 
            shuffle=False
        )
        return X_new, y_new

"""
Опис класу DataGenerator:

Цей клас відповідає за генерацію синтетичних даних для задачі кластеризації за допомогою бібліотеки sklearn.
Він ініціалізує параметри кластерів (центри та стандартне відхилення)
і створює базовий набір даних при створенні об'єкта.
Ключова особливість — наявність методу generate_custom_size, 
який дозволяє динамічно створювати набори даних довільного розміру.
"""

class DataVisualizer:
    def __init__(self, X, y_true):
        self.X = X
        self.y_true = y_true

    def plot_initial_data(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y_true, s=10, cmap='viridis')
        plt.title('1. Початкові дані (справжні мітки)')
        plt.xlabel('Ознака 1 (X1)')
        plt.ylabel('Ознака 2 (X2)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

"""
Опис класу DataVisualizer:

Цей клас відповідає за первинну візуалізацію даних. 
Він будує діаграму розсіювання для вхідних точок, 
розфарбовуючи їх відповідно до відомих "істинних" міток. 
"""

class ModelBuilder:
    def __init__(self, min_samples=10, min_cluster_size=20, xi=0.1, max_eps=5, metric='euclidean'):
        print("\n=== Пункт 2: побудова моделей кластеризації ===")
        
        self.params = {
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size,
            'xi': xi,
            'max_eps': max_eps,
            'metric': metric,  
        }
        self.model = None

    def build_model(self, verbose=True):
        self.model = OPTICS(**self.params)
        
        if verbose:
            print(f"Модель OPTICS створено з параметрами:\n {self.params}")
            
        return self.model

"""
Опис класу ModelBuilder:

Цей клас слугує обгорткою для алгоритму кластеризації OPTICS з бібліотеки.
Він інкапсулює налаштування гіперпараметрів моделі (таких як min_samples, xi, metric...) 
та надає метод build_model для створення нового екземпляра алгоритму.
Це дозволяє централізовано керувати конфігурацією та спрощує створення нових об'єктів моделі, 
що особливо корисно при оцінці швидкодії.
"""

class ClusterExecutor:
    def __init__(self, model, X_data):
        self.model = model
        self.X = X_data

    def run_clustering(self):
        print("\n=== Пункт 3: виконання кластеризації ===")
        
        if self.model is None:
            print("Помилка: модель не була передана виконавцю.")
            return None, 0.0

        start_time = time.perf_counter()
        labels = self.model.fit_predict(self.X)
        end_time = time.perf_counter()

        execution_time = end_time - start_time

        n_clusters_ = len(np.unique(labels[labels != -1]))
        n_noise_ = list(labels).count(-1)

        print(f"OPTICS: знайдено {n_clusters_} кластер(ів).")
        print(f"OPTICS: кількість шумових точок: {n_noise_}.")
        print(f"\n Час кластеризації: {execution_time:.4f} секунд.")

        return labels, execution_time

"""
Опис класу ClusterExecutor:

Цей клас відповідає за безпосереднє виконання процесу кластеризації. 
Він приймає вже налаштовану модель та дані, запускає навчання 
і фіксує час виконання цієї операції.
Після завершення обчислень клас аналізує отримані мітки, 
підраховує кількість знайдених кластерів і шумових точок, 
та виводить цю статистику разом із часом виконання в консоль.
"""

class ClusterVisualizer:
    def __init__(self, model, X_data):
        self.model = model
        self.X = X_data
        self.labels = model.labels_

    def _generate_color_map(self):
        unique_labels = sorted(set(self.labels))
        n_clusters = len([l for l in unique_labels if l != -1])

        cluster_colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
        
        color_map = {}
        color_iter = iter(cluster_colors)

        for lab in unique_labels:
            if lab == -1:
                color_map[lab] = (0, 0, 0, 1)  
            else:
                color_map[lab] = next(color_iter)

        return color_map

    def plot_reachability(self):
        if self.labels is None:
            print("Помилка: немає міток кластерів.")
            return

        ordering = self.model.ordering_
        reachability = self.model.reachability_[ordering]
        labels_ordered = self.labels[ordering]

        color_map = self._generate_color_map()

        plt.figure(figsize=(12, 6))
        plt.title("Графік досяжності")
        plt.xlabel("Порядок точок")
        plt.ylabel("Відстань досяжності")

        unique_labels = sorted(set(self.labels))
        
        for lab in unique_labels:
            mask = labels_ordered == lab
            plt.plot(
                np.where(mask)[0], 
                reachability[mask], 
                '.', 
                markersize=4,
                color=color_map[lab],
                label=f"Кластер {lab}" if lab != -1 else "Шум")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_clusters(self):
        if self.labels is None:
            print("Помилка: немає міток кластерів.")
            return

        color_map = self._generate_color_map()
        point_colors = [color_map[label] for label in self.labels]

        n_clusters = len([l for l in set(self.labels) if l != -1])

        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=point_colors, s=12)
        plt.title(f"Результат кластеризації (Кластерів: {n_clusters})")
        plt.xlabel("Ознака 1 (X1)")
        plt.ylabel("Ознака 2 (X2)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        self.plot_reachability()
        self.plot_clusters()

"""
Опис класу ClusterVisualizer:

Цей клас відповідає за візуалізацію результатів роботи алгоритму OPTICS.
Він генерує два типи графіків:
1. Графік досяжності - специфічна діаграма для OPTICS, 
що показує відстані досяжності для впорядкованих точок. 
Це дозволяє візуально оцінити щільність кластерів (вони виглядають як западини на графіку).

2. Діаграма розсіювання - класичне відображення точок у 2D просторі, 
де кожен кластер забарвлений у свій колір, а шумові точки виділені чорним.

Також клас містить допоміжний метод для генерації кольорової палітри.
"""

class PerformanceEvaluator:
    def __init__(self, data_generator, model_builder):
        print("\n=== Пункт 5: оцінка швидкодії ===")
        self.data_gen = data_generator
        self.builder = model_builder

    def evaluate(self, sizes_list):
        print(f"{'Кількість точок ':<15} | {'Час(сек) ':<10}")
        print("-" * 30)

        model = self.builder.build_model(verbose=False)

        execution_times = []

        for n_samples in sizes_list:
            X, _ = self.data_gen.generate_custom_size(n_samples)
            
            start = time.perf_counter()
            model.fit_predict(X)
            exec_time = time.perf_counter() - start

            execution_times.append(exec_time)

            print(f"{n_samples:<15} | {exec_time:<10.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(sizes_list, execution_times, marker='o', linestyle='-', color='b', linewidth=2)
        
        plt.title("Залежність часу кластеризації від кількості точок")
        plt.xlabel("Кількість точок")
        plt.ylabel("Час виконання")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

"""
Опис класу PerformanceEvaluator:

Цей клас призначений для оцінки швидкодії алгоритму кластеризації.
Він автоматизує процес стрес-тестування: генерує набори даних різного розміру
та вимірює час, необхідний моделі для їх обробки.
Результати виводяться у вигляді таблиці в консоль для точного аналізу та відображаються на графіку,
що дозволяє візуально оцінити, як зростає час обчислень при збільшенні обсягу даних.
"""

class AlternativeModelRunner:
    def __init__(self, X, y_true=None):
        self.X = X
        self.results = {}  
        print("\n=== Пункт 6: побудова альтернативних моделей ===")

    def _save_result(self, name, model, labels, exec_time):
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        print(f"  {name}: {n_clusters} кластери, шум={n_noise}, час={exec_time:.4f} сек.")

        self.results[name] = {
            "model": model,
            "labels": labels,
            "time": exec_time,
            "params": model.get_params(),
            "clusters": n_clusters,
            "noise": n_noise}

    def _fit_and_save(self, model, name):
        start = time.time()
        labels = model.fit_predict(self.X)
        t = time.time() - start
        self._save_result(name, model, labels, t)

    def run_experiments(self):
        EPS = 50.0 
        
        print("\n1. Експерименти з min_samples:")

        for ms in [5, 10, 20, 30]:
            model = OPTICS(
                min_samples=ms, 
                min_cluster_size=10, 
                xi=0.15,            
                max_eps=EPS, 
                metric='manhattan')
            self._fit_and_save(model, f"min_samples={ms}")

        print("\n2. Експерименти з min_cluster_size:")
        for cs in [10, 30, 60, 100]:
            model = OPTICS(
                min_samples=10,    
                min_cluster_size=cs, 
                xi=0.15, 
                max_eps=EPS, 
                metric='manhattan')
            self._fit_and_save(model, f"min_cluster_size={cs}")

        print("\n3. Експерименти з xi:")
        for xi in [0.01, 0.05, 0.15, 0.25]:
            model = OPTICS(
                min_samples=10, 
                min_cluster_size=10, 
                xi=xi, 
                max_eps=EPS, 
                metric='manhattan')
            self._fit_and_save(model, f"xi={xi}")

        print("\n4. Експерименти з метриками:")
        for metric in ['euclidean', 'manhattan']:
            model = OPTICS(
                min_samples=10, 
                min_cluster_size=30, 
                xi=0.15, 
                max_eps=EPS, 
                metric=metric)
            self._fit_and_save(model, f"metric={metric}")

        print("\n5. Комбінації параметрів:")
        combos = [
            (10, 10, 0.15, 'manhattan'), 
            (20, 30, 0.05, 'manhattan'), 
            (12, 15, 0.1, 'manhattan'),   
            (10, 15, 0.1, 'euclidean'),  
            (12, 20, 0.12, 'manhattan'),  
            (14, 20, 0.15, 'euclidean'),]

        for ms, cs, xi, metric in combos:
            model = OPTICS(
                min_samples=ms, 
                min_cluster_size=cs, 
                xi=xi, 
                max_eps=EPS, 
                metric=metric)
            name = f"comb(ms={ms}, cs={cs}, xi={xi}, metric={metric})"
            self._fit_and_save(model, name)


        print(f"\n Створено {len(self.results)} альтернативних моделей")
        return self.results

"""
Опис класу AlternativeModelRunner:

Цей клас автоматизує процес дослідження впливу гіперпараметрів на якість кластеризації алгоритмом OPTICS.
Принцип роботи методу `run_experiments`:
1.  Виконує серію ізольованих експериментів, змінюючи по черзі один з ключових параметрів, залишаючи інші фіксованими.
2.  Виконує тестування специфічних комбінацій параметрів (змінюючи декілька налаштувань одночасно), 
    щоб знайти оптимальний баланс для складних даних.
3.  Для кожної моделі вимірює час виконання та проводить первинний аналіз результатів (підрахунок кількості кластерів та шумових точок).
4.  Зберігає повний стан кожного експерименту (навчену модель, отримані мітки, час виконання) у словник `self.results`.
"""

class MetricsCalculator:
    def __init__(self, X_data, y_true, results):
        self.X = X_data
        self.y_true = y_true
        self.results = results

    def calculate_all_metrics(self):
        print("\n=== Пункт 7: розрахунок метрик якості ===\n")

        metrics_output = {}
        table_rows = []

        for name, info in self.results.items():
            labels = info["labels"]
            unique_labels = set(labels)

            clean_clusters = unique_labels - {-1}
            n_clusters_clean = len(clean_clusters)

            estimated_clusters = len(unique_labels)

            noise_ratio = list(labels).count(-1) / len(labels)

            ari = adjusted_rand_score(self.y_true, labels)
            ami = adjusted_mutual_info_score(self.y_true, labels)

            if n_clusters_clean >= 2:
                try:
                    sil = silhouette_score(self.X, labels)
                except Exception:
                    sil = None
            else:
                sil = None

            metrics_output[name] = {
                "clusters_clean": n_clusters_clean,
                "estimated_clusters": estimated_clusters,
                "noise_ratio": noise_ratio,
                "ARI": ari,
                "AMI": ami,
                "Silhouette": sil}

            table_rows.append([
                name,
                n_clusters_clean,
                estimated_clusters,
                f"{ari:.4f}",
                f"{ami:.4f}",
                f"{sil:.4f}" if sil is not None else "None",
                f"{noise_ratio:.3f}"])

        print(tabulate(
            table_rows,
            headers=["Model", "Clusters(no-noise)", "EstimatedClusters", "ARI", "AMI", "Silhouette", "NoiseRatio"],
            tablefmt="github"))

        print("\n Метрики розраховано \n")

        self._print_top_5(metrics_output)
        return metrics_output

    def _print_top_5(self, metrics_output):
        print("ТОП-5 найкращих \n")

        ranked = []

        for name, stats in metrics_output.items():
            k = stats["clusters_clean"]

            if k < 2:
                continue

            ari = stats["ARI"]
            sil = stats["Silhouette"]
            noise = stats["noise_ratio"]

            sil_score = sil if sil is not None else -0.5

            score = 0.6 * ari + 0.4 * sil_score - noise

            ranked.append((score, name, stats))

        ranked.sort(reverse=True, key=lambda x: x[0])

        for i, (score, name, stats) in enumerate(ranked[:5], 1):
            print(f"#{i} Score={score:.4f} | ARI={stats['ARI']:.4f} | "
                  f"Sil={stats['Silhouette'] if stats['Silhouette'] is not None else 'None'} "
                  f"| Clusters={stats['clusters_clean']} | Noise={stats['noise_ratio']:.3f}")
            print(f"    Параметри моделі: {name}")
            print("-" * 60)

"""
Опис класу MetricsCalculator:

Цей клас відповідає за обчислення та аналіз метрик якості кластеризації. 
Він використовує три основні методи з бібліотеки sklearn:
1. `adjusted_rand_score` (ARI) — для порівняння знайдених кластерів з еталонними мітками.
2. `adjusted_mutual_info_score` (AMI) — для оцінки взаємної інформації між розбиттями.
3. `silhouette_score` — для оцінки щільності кластерів (внутрішня метрика).

Клас також автоматично підраховує кількість шуму  і формує рейтинг моделей, використовуючи зважену формулу, 
що балансує точність (ARI), якість структури (Silhouette) та кількість втрачених даних (Noise).
"""

class StabilityAnalyzer:
    def __init__(self, X_data, original_labels, model_params, n_permutations=10):
        self.X = X_data
        self.original_labels = original_labels
        self.model_params = model_params
        self.n_permutations = n_permutations

        print("\n=== Пункт 8: аналіз стабільності кластеризації при зміні порядку ===")


    def check_permutation_stability(self):
        ari_scores = []
        ami_scores = []
        equal_count = 0

        print(f"\nАналізуємо {self.n_permutations} випадкових перестановок порядку точок...\n")

        for i in range(self.n_permutations):
            n_samples = self.X.shape[0]
            indices = np.arange(n_samples)
            rng = np.random.default_rng(seed=42 + i)
            rng.shuffle(indices)

            X_shuffled = self.X[indices]

            model = OPTICS(**self.model_params)
            labels_shuffled = model.fit_predict(X_shuffled)

            inverse_indices = np.argsort(indices)
            labels_unshuffled = labels_shuffled[inverse_indices]

            ari = adjusted_rand_score(self.original_labels, labels_unshuffled)
            ami = adjusted_mutual_info_score(self.original_labels, labels_unshuffled)

            ari_scores.append(ari)
            ami_scores.append(ami)

            if np.array_equal(self.original_labels, labels_unshuffled):
                equal_count += 1

            print(f"Перестановка {i+1}: ARI={ari:.4f}, AMI={ami:.4f}")

        mean_ari = np.mean(ari_scores)
        mean_ami = np.mean(ami_scores)

        print(f"Середній ARI між оригінальною та переставленими кластеризаціями: {mean_ari:.4f}")
        print(f"Середній AMI: {mean_ami:.4f}")

        if equal_count == self.n_permutations:
            print("Так, розбиття є абсолютно стабільним незалежно від порядку даних.")
        elif mean_ari > 0.90:
            print("Розбиття загалом стабільне: перестановки майже не змінюють структуру кластерів.")
        elif mean_ari > 0.50:
            print("Середня стабільність: порядок даних впливає, але не критично.")
        elif mean_ari > 0.10:
            print("Низька стабільність: порядок суттєво змінює результат OPTICS.")
        else:
            print("Розбиття абсолютно нестабільне: OPTICS щоразу дає різні кластери.")

"""
Опис класу StabilityAnalyzer:

Цей клас виконує перевірку алгоритму на чутливість до порядку вхідних даних.
Принцип роботи методу `check_permutation_stability`:
1.  Генерує випадкові перестановки індексів масиву даних.
2.  Перемішує вхідні дані згідно з цими індексами.
3.  Заново проводить кластеризацію на перемішаних даних, використовуючи ті самі гіперпараметри моделі.
4.  Критично важливий крок: відновлює порядок отриманих міток до початкового стану, використовуючи np.argsort.
5.  Порівнює нові мітки з оригінальними (отриманими на неперемішаних даних) за допомогою метрик ARI та AMI.

Якщо середній ARI близький до 1.0, це означає, що алгоритм працює стабільно і порядок рядків у датасеті не впливає на кінцевий результат.
"""

if __name__ == "__main__":
    # 0. Генерація даних
    generator = DataGenerator()
    X, y_true = generator.get_data()

    # 1. Візуалізація початкових даних
    p1_visualizer = DataVisualizer(X, y_true)
    p1_visualizer.plot_initial_data() 

    # 2. Побудова моделі
    p2_builder = ModelBuilder()
    optics_model = p2_builder.build_model()
    model_params = p2_builder.params

    # 3. Кластеризація
    p3_executor = ClusterExecutor(model=optics_model, X_data=X)
    cluster_labels, exec_time = p3_executor.run_clustering()

    # 4. Візуалізація результатів
    visualizer = ClusterVisualizer(optics_model, X)
    visualizer.plot_all()
    
    # 5. Оцінка швидкодії
    evaluator = PerformanceEvaluator(generator, p2_builder)
    sizes = [5000, 10000, 50000, 100000]
    evaluator.evaluate(sizes)

    # 6. Альтернативні моделі
    runner = AlternativeModelRunner(X, y_true)
    results = runner.run_experiments()

    # 7. Розрахунок метрик
    p7_calculator = MetricsCalculator(
       X_data=X, 
       y_true=y_true, 
       results=results)
    p7_calculator.calculate_all_metrics()

    # 8. Аналіз стабільності
    p8_analyzer = StabilityAnalyzer(
        X_data=X, 
        original_labels=cluster_labels, 
        model_params=model_params)
    p8_analyzer.check_permutation_stability()
