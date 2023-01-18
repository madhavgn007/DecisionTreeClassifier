import time
import psutil
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from DecisionTreeClassifier import DecisionTree
import pandas as pd

def analytics(X, y, classifier=None):
    test_size = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    columns=['Test-Train Split (%)','Max_depth','Min_Sample_Split','Time taken to train classifier (sec)', 
    'Memory used (MB)', 'CPU used', 'Accuracy Score', 'Precision Score', 'F1-Score', 'Recall Score']
    result_df = pd.DataFrame(columns=columns)
    for ts in test_size:
        for md in max_depth:
            for min_split in min_samples_split:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=2023)
                current_process = psutil.Process()
                mem_before = current_process.memory_info().rss
                cpu_before = current_process.cpu_percent()
                start_time = time.time()
                if classifier == 'own':
                    clf = DecisionTree(max_depth=md, min_samples_split=min_split)
                else:
                    clf = DecisionTreeClassifier(max_depth=md, min_samples_split=min_split)
                clf.fit(X_train, y_train)
                end_time = time.time()
                time.sleep(0.01)
                mem_after = current_process.memory_info().rss
                cpu_after = current_process.cpu_percent()
                time_elapsed = end_time - start_time
                mem_used = (mem_after - mem_before) / (1024 ** 2)
                cpu_used = cpu_after - cpu_before
                y_pred = clf.predict(X_test)
                temp_df = pd.DataFrame([[ts*100, md, min_split, time_elapsed, mem_used, cpu_used, 
                accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='weighted'), 
                f1_score(y_test, y_pred, average='weighted'), recall_score(y_test, 
                y_pred, average='weighted')]], columns=columns)
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
    return result_df