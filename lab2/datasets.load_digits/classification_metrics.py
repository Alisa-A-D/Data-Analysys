import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Клас 9: розрахунок метрик класифікації без текстового виводу
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

        metrics['report'] = classification_report(y_true, y_pred, target_names=[str(c) for c in self.classes])

        metrics['cm'] = confusion_matrix(y_true, y_pred)

        y_true_bin = label_binarize(y_true, classes=self.classes)

        metrics['fpr'], metrics['tpr'], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        metrics['roc_auc'] = auc(metrics['fpr'], metrics['tpr'])

        metrics['precision'], metrics['recall'], _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
        
        return metrics

    def evaluate_all(self):
        for name, model in self.models.items():
            y_pred_train = self.predictions[name]["y_pred_train"]
            y_pred_test = self.predictions[name]["y_pred_test"]

            y_scores_train = model.predict_proba(self.X_train)
            y_scores_test = model.predict_proba(self.X_test)

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
                     label='TRAIN PR-крива')
            ax2.plot(test_metrics['recall'], test_metrics['precision'], color='green', lw=2,
                     label='TEST PR-крива')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('PR-крива (Train vs Test)')
            ax2.legend(loc="lower left")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
            plt.show()
