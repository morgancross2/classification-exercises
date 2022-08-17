# standard ds imports
import numpy as np
import pandas as pd

# for model evaluation
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

'''
This file contains:
    BINARY TARGET MODELS:
        Classification:
            - binary_decisiontree_data(X_train, y_train)
            - binary_randomforest_data(X_train, y_train)
            - binary_KNN_data(X_train, y_train, weight)
            - binary_logisticregression_data(X_train, y_train, penalty='l2')
        
        Regression:
        
        Clustering:
        
        Time Series:
        
'''


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
        FNR = round(((FN)/(TP+FN))*100,2)
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
            FNR = round(((FN)/(TP+FN))*100,2)
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


def binary_KNN_data(X_train, y_train, weight):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        weights = uniform or distance as a string
        
    Returns a DataFrame running K Nearest Neighbor models with confusion matrix data
    from 1 to 20 neighbors.
    '''
    models = {}
    z = 1
    for i in range(1,21):
        model = KNeighborsClassifier(n_neighbors=i, weights=weight)
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
        FNR = round(((FN)/(TP+FN))*100,2)
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


def binary_logisticregression_data(X_train, y_train, penalty='l2'):
    '''
    ONLY FOR A BINARY TARGET. 
    (Built on titanic data where target was survived (0 or 1))
    
    This function takes in:
        X_train = train dataset minus target series as DataFrame
        y_train = target variable column as a series
        penalty = default to l2, can apply l1, l2, none, or elasticnet
        
    Returns two DataFrames (models and odds):
        Models df - Creates Logistic Regression models with confusion matrix data
                    from .01 to 1000 C (increment logrithmically)
    
        Odds df - Displays odds of X_train features
                a. If the coefficient (odds) is 1 or close to 1, this means odds of 
                    being in class '1' (positive class) is same or close to being in class 
                    '0' (negative class). This means the feature with this coefficient is 
                    not a big driver for the target variable in this particular model.
                b. If the coefficient value is < 1 , that implies that increase in 
                    value of that feature will decrease the odds that target variable is 
                    in positive class.
    '''
    models = {}
    odds = {}
    z = 1
    span = [.01, .1, 1, 10, 100, 1000]
    for i in range(0,6):
        model = LogisticRegression(C=span[i])
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
        FNR = round(((FN)/(TP+FN))*100,2)
        percision = round((TP/(TP+FP))*100,2)
        f1 = round((met.f1_score(y_train, model.predict(X_train)))*100,2)
        sp = TP + FN
        sn = FP + TN
        
        model_name = 'logit ' + str(z)
        models[model_name] = {'C': span[i],
                        'accuracy' : acc.astype(str)+'%', 
                        'recall_tpr': TPR.astype(str)+'%', 
                        'specificity_tnr': TNR.astype(str)+'%', 
                        'fpr': FPR.astype(str)+'%', 
                        'fnr': FNR.astype(str)+'%', 
                        'percision': percision.astype(str)+'%',
                        'f1': f1.astype(str)+'%',
                        'support_pos': sp,
                        'support_neg': sn}
        odds[model_name] = model.coef_[0]
        z += 1
    models = pd.DataFrame(models).T
    odds = np.exp(pd.DataFrame(odds, index=X_train.columns)).T
    odds['C'] = models['C']
        
    return models, odds