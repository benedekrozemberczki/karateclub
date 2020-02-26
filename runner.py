import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from karateclub import GraphReader, LaplacianEigenmaps
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression



def tester(y, X, name):
    aucs = []
    for i in tqdm(range(100)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        downstream_model = LogisticRegression(random_state=0,solver ="saga").fit(X_train, y_train)
        y_hat = downstream_model.predict_proba(X_test)[:, 1]
        if name == "facebook":
            y_hat = downstream_model.predict_proba(X_test)
            new_target = np.zeros((y_test.size, y_test.max()+1))
            new_target[np.arange(y_test.size),y_test] = 1
            y_test = new_target
        auc = roc_auc_score(y_test, y_hat)
        print(auc)
        
        aucs.append(auc)
    print(round(np.mean(aucs),3))
    print(round(np.std(aucs),3))


def runner(name):
    print("-------------------------------------------------")
    print(name)
    print("-------------------------------------------------")
    
    reader = GraphReader(name)
    graph = reader.get_graph()
    features = reader.get_features()
    y = reader.get_target()
    print("LaplacianEigenmaps")
    model = LaplacianEigenmaps()
    model.fit(graph)
    X = model.get_embedding()
    tester(y, X, name)
    #print("-------------------------------------------------")
    #print("DeepWalk")
    #model = DeepWalk()
    #model.fit(graph)
    #X = model.get_embedding()
    #tester(y, X)
    #print("-------------------------------------------------")
    #print("Walklets")
    #model = Walklets()
    #model.fit(graph)
    #X = model.get_embedding()
    #tester(y, X)




runner("facebook")
