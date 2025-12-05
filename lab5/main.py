from data_visualizer import FriedmanVisualizer
from data_splitter import FriedmanDataSplitter
from ensemble_experiment import GradientBoostingExperiment

if __name__ == "__main__":
     # Пункт 1: генерація та візуалізація даних
    handler = FriedmanVisualizer()
    handler.plot_all_features()

    # Пункт 2: розбиття даних
    splitter = FriedmanDataSplitter(handler.X, handler.y)
    splitter.split(test_size=0.2, val_size=0.2)
    splitter.show_split_sizes()

    # Пункт 3: побудова ансамблів
    experiment = GradientBoostingExperiment(
        splitter.X_train, splitter.y_train,
        splitter.X_val, splitter.y_val,
        splitter.X_test, splitter.y_test)
    best_params = experiment.run()
