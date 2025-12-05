from sklearn.model_selection import train_test_split

class FriedmanDataSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

    def split(self, test_size=0.2, val_size=0.2, random_state=42):
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        if (1 - test_size) == 0:
            raise ValueError("Розмір вибірки не може бути 1.0")
            
        val_ratio = val_size / (1 - test_size)
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state)
        return True

    def show_split_sizes(self):
        print("\n === Пункт 2: розбиття даних === \n" )
        print("Результати розбиття:")
        print(f"Повна вибірка: {len(self.X)}")
        print(f"Навчальна:   {len(self.X_train)}")
        print(f"Перевірочна:  {len(self.X_val)}")
        print(f"Тестова: {len(self.X_test)}")