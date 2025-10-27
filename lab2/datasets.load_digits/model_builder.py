from sklearn.linear_model import LogisticRegression

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