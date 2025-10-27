from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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