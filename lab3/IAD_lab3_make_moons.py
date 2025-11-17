from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
recall_score, f1_score, roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Step 1: Visualization
def data_visualizer(X, Y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
    plt.title("Step 1: Data set 'moons'")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
#Step 2: Splitt data    
def data_splitter(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.3, random_state = 42)
    print(f"Size train: {X_train.shape}, size test: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test
#Step 3: Building regression models    
def models_builder(X_train, Y_train):
    models = {}
    labels = [(50, "50_no_scale","50_scale"),(100, "100_no_scale","100_scale"),(150, "150_no_scale","150_scale")]
    temp = StandardScaler()
    temp.fit(X_train)
    X_train_trform = temp.transform(X_train)
    for l1, l2, l3 in labels:
        models[l2] = MLPClassifier(hidden_layer_sizes=(l1,), max_iter=200)
        models[l3] = MLPClassifier(hidden_layer_sizes=(l1,), max_iter=200)
        models[l2].fit(X_train, Y_train)
        models[l3].fit(X_train_trform, Y_train)
        print(f"Model '{l2}' trained.\nModel '{l3}' trained.")
    return models
#Step 4: Models visualization
def model_visualizer(X_train, models):
    for name in models.keys():
        temp = models[name].predict_proba(X_train)
        df = np.empty(len(temp))
        #print(temp)
        for i in range(len(temp)):
            if temp[i][0] > temp[i][1]:
                df[i] = -(temp[i][0] - 0.5)
            else:
                df[i] = temp[i][1] - 0.5
        #print(f"Model '{name}' df:{df}")
        plt.scatter(X_train[:,0], X_train[:,1], c=df, cmap='seismic', edgecolors='k')            
        plt.title(name)
        plt.show()
#Step 5: Predictions
def predict(models, X_train, X_test):
    results = {}
    for name, model in models.items():
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_test)
        results[name]={"y_pred_train": Y_pred_train, "y_pred_test": Y_pred_test}
        print(f"Prediction for model '{name}' successful.")
    return results
#Step 6: Overfitting
def overfitting_estimation(results, Y_train, Y_test): 
    for name in results.keys():
         acc_train = accuracy_score(Y_train,results[name]["y_pred_train"])
         acc_test = accuracy_score(Y_test,results[name]["y_pred_test"])
         print(f"..... Model: {name} .....\nTrain accuracy: {acc_train}\nTest accuracy: {acc_test}\nDifference: {acc_train-acc_test}")
         if acc_train-acc_test > 0.05:
             print("Model is overfitted!\n") 
         else:
             print("No overfitting detected!\n") 
#Step 7: Posterior probabilities
def posterior_prob(models, predictions, X_test, Y_test):
    for name, model in models.items():
        probs = np.array(model.predict_proba(X_test[60:70]))
        df_probs = pd.DataFrame(np.round(probs, 4), columns=[f"Class {i}" for i in range(probs.shape[1])])
        df_probs['Prediction']=np.array(predictions[name]["y_pred_test"][60:70])
        df_probs['Original class']=np.array(Y_test[60:70])
        print(f"Model {name}\n{df_probs}\n")
#Step 8: Decision boundaries
def decision_boundaries(X, Y, models):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200))
    for name in models.keys():
        Z = models[name].predict(np.c_[xx.ravel(),yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, s=60, edgecolors='k')
        plt.title(f"Decision boundaries for model '{name}'")
    plt.show()    
#Step 9: Quality criteria
def quality_criteria(models, predictions, X_train, X_test, Y_train, Y_test):
    labels = [("Train set", Y_train, "y_pred_train", X_train), ("Test set", Y_test, "y_pred_test", X_test)]
    for name in predictions.keys():
         metrics = {} 
         for l1, l2, l3, l4 in labels:
             metrics[l1]={"Confusion matrix": confusion_matrix(l2, predictions[name][l3]),
                          "Precision": precision_score(l2, predictions[name][l3]),
                          "Recall": recall_score(l2, predictions[name][l3]),
                          "F1 score": f1_score(l2, predictions[name][l3]),
                          "AUC score": roc_auc_score(l2, models[name].predict_proba(l4)[:,1])}
             PrecisionRecallDisplay.from_estimator(models[name], l4, l2)
             plt.title(f"PR_curve for '{name}' on {l1}")
             RocCurveDisplay.from_estimator(models[name], l4, l2)
             plt.title(f"ROC_curve for '{name}' on {l1}")
             plt.show()
         df_metrics = pd.DataFrame(metrics)
         print(f"..... Metrics for model: {name} .....\n{df_metrics}\n")
#Step 13: Smaller splits of X/Y sets
def smaller_splits(X, Y):
    sizes=([0.1, 0.2])
    print(f"Smaller train sets:")
    for size in sizes:
        X_new_train, X_new_test, Y_new_train, Y_new_test = train_test_split(X, Y, train_size = size, random_state = 42)
        print(f"Size train: {X_new_train.shape}, size test: {X_new_test.shape}")
        models = models_builder(X_new_train, Y_new_train)
        predictions = predict(models, X_new_train, X_new_test)
        overfitting_estimation(predictions, Y_new_train, Y_new_test)

  if __name__ == "__main__":
    X, Y = make_moons(n_samples=200, noise=0.2, random_state=42)
    data_visualizer(X, Y)
    print(f"Step 2: Splitt data")
    X_train, X_test, Y_train, Y_test = data_splitter(X, Y)
    print(f"\nStep 3: Building models")
    models = models_builder(X_train, Y_train)
    model_visualizer(X_train, models) 
    print(f"\nStep 5: Predictions")
    predictions = predict(models, X_train, X_test)
    print(f"\nStep 6: Overfitting estimation")
    overfitting_estimation(predictions, Y_train, Y_test)
    print(f"Step 7: Posterior probabilities")
    posterior_prob(models, predictions, X_test, Y_test)
    decision_boundaries(X, Y, models) 
    print(f"\nStep 9: Quality criteria")
    quality_criteria(models, predictions, X_train, X_test, Y_train, Y_test)
    
    print(f"\nStep 13: Smaller splitts of X/Y sets")
    smaller_splits(X, Y)
