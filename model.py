# standard ds imports
import numpy as np
import pandas as pd

# for model evaluation
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def binary_decisiontree_data(X_train, y_train):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    a
    This function takes in:
        X_train = train dataset minus target series as DataFrame 
        y_train = target variable column as a series
        
    Returns a DataFrame running decision tree models from 2 to 7 depth with confusion matrix data.
    '''
    trees = {}
    z = 1
    for i in range(2,8):
        tree = DecisionTreeClassifier(max_depth=i)
        tree.fit(X_train, y_train)

        cm = met.confusion_matrix(y_train, tree.predict(X_train))
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]

        acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
        TPR = round((TP/(TP+FN))*100,2)
        TNR = round(((TN)/(FP+TN))*100,2)
        FPR = round((FP / (FP + TN))*100,2)
        FNR = round(((FN)/(TN+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, tree.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN

        model_name = 'tree '+str(z)
        trees[model_name] = {'max_depth': i,
                         'accuracy' : acc.astype(str)+'%', 
                         'recall_TPR': TPR.astype(str)+'%', 
                         'specificity_TNR': TNR.astype(str)+'%', 
                         'FPR': FPR.astype(str)+'%', 
                         'FNR': FNR.astype(str)+'%', 
                         'percision': percision.astype(str)+'%',
                         'f1': f1.astype(str)+'%',
                         'support_pos': sp,
                         'support_neg': sn}
        z+=1
    return pd.DataFrame(trees).T


def binary_randomforest_data(X_train, y_train):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        
    Returns a DataFrame running random forest models with confusion matrix data
    from 2 to 9 depth and 1 to 4 min sample leaf.
    '''
    forests = {}
    z = 1
    for i in range(2,10):
        for x in range(1,5):
            forest = RandomForestClassifier(max_depth=i, min_samples_leaf=x, random_state=123)
            forest.fit(X_train, y_train)
    
            cm = met.confusion_matrix(y_train, forest.predict(X_train))
            TP = cm[1,1]
            FP = cm[0,1]
            TN = cm[0,0]
            FN = cm[1,0]
    
            acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
            TPR = round((TP/(TP+FN))*100,2)
            TNR = round(((TN)/(FP+TN))*100,2)
            FPR = round((FP / (FP + TN))*100,2)
            FNR = round(((FN)/(TN+FN))*100,2)
            percision = round((TP/(TP+FP))*100,2)
            f1 = round((met.f1_score(y_train, forest.predict(X_train)))*100,2)
            sp = TP + FN
            sn = FP + TN
            
            model_name = 'forest ' + str(z)
            forests[model_name] = {'max_depth' : i,
                         'min_samples_leaf' : x,
                         'accuracy' : acc.astype(str)+'%', 
                         'recall_TPR': TPR.astype(str)+'%', 
                         'specificity_TNR': TNR.astype(str)+'%', 
                         'FPR': FPR.astype(str)+'%', 
                         'FNR': FNR.astype(str)+'%', 
                         'percision': percision.astype(str)+'%',
                         'f1': f1.astype(str)+'%',
                         'support_pos': sp,
                         'support_neg': sn}
            z += 1
    return pd.DataFrame(forests).T

def binary_KNN_data(X_train, y_train):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        
    Returns a DataFrame running K Nearest Neighbor models with confusion matrix data
    from 5 to 35 (increment by 5) neighbors.
    '''
    models = {}
    z = 1
    for i in range(5,30,5):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        
        cm = met.confusion_matrix(y_train, model.predict(X_train))
        TP = cm[1,1]
        FP = cm[0,1]
        TN = cm[0,0]
        FN = cm[1,0]
        
        acc = round(((TP+TN)/(TP+FP+FN+TN))*100,2)
        TPR = round((TP/(TP+FN))*100,2)
        TNR = round(((TN)/(FP+TN))*100,2)
        FPR = round((FP / (FP + TN))*100,2)
        FNR = round(((FN)/(TN+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, model.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN
        
        model_name = 'knn ' + str(z)
        models[model_name] = {'neighbors': i,
                        'accuracy' : acc.astype(str)+'%', 
                        'recall_tpr': TPR.astype(str)+'%', 
                        'specificity_tnr': TNR.astype(str)+'%', 
                        'fpr': FPR.astype(str)+'%', 
                        'fnr': FNR.astype(str)+'%', 
                        'percision': percision.astype(str)+'%',
                        'f1': f1.astype(str)+'%',
                        'support_pos': sp,
                        'support_neg': sn}
        z += 1
    return pd.DataFrame(models).T