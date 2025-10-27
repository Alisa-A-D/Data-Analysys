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
