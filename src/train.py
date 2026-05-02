from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    params = {'C': [0.1, 1, 5]}
    
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    return grid.best_estimator_