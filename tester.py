from karateclub.dataset import GraphSetReader

reader = GraphSetReader("reddit10k")

graphs = reader.get_graphs()
y = reader.get_target()[0:200]

from karateclub.graph_embedding import IGE

model = IGE()
model.fit(graphs[0:200])
X = model.get_embedding()

from sklearn.model_selection import train_test_split

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
print(y)
downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('AUC: {:.4f}'.format(auc))
