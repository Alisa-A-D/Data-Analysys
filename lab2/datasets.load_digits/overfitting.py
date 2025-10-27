from sklearn.metrics import accuracy_score

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