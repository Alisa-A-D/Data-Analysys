import numpy as np
import pandas as pd

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