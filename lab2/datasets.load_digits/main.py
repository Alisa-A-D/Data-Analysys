import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression

from digits_visualizer import DigitsVisualizer
from data_splitter import DigitsDataSplitter
from model_builder import LogisticModelsBuilder
from model_visualizer import LogisticModelVisualizer
from predictor import ModelPredictor
from overfitting import OverfittingChecker
from probabilities import PosteriorProbabilitiesCalculator
from decision_boundary_visualizer import DecisionBoundaryVisualizer
from classification_metrics import ClassificationEvaluator
from grid_search import ModelGridSearcher
from learning_curve import TrainingSizeEvaluator

warnings.simplefilter(action='ignore', category=FutureWarning)

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