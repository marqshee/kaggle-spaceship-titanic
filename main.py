import pandas as pd
from preprocessor import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

df_train = pd.read_csv('train.csv')
# df_train[df_train.isna().any(axis=1)]
# print(df_train.head())

# Test preprocessor on an analytics base table
abt = df_train.copy()

# Clean up the data
df_abt = preprocessing(abt)

# Drop target value trying to predict (Transported) and PassengerId
X = df_abt.drop(['Transported', 'PassengerId'], axis=1)

# Convert catagorical vars into dummy/indicator vars
X = pd.get_dummies(X)

# Transported is the target to predict
y = df_abt['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=1234)),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1234))
}

grid = {
    'rf': {
        'randomforestclassifier__n_estimators': [100, 200, 300]
    },
    'gb': {
        'gradientboostingclassifier__n_estimators': [100, 200, 300]
    }
}

# Model
fit_models = {}
for algo, pipeline in pipelines.items():
    print(f'Training the {algo} model...')
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, cv=10)
    model.fit(X_train, y_train)
    fit_models[algo] = model
    
# Evaluate model performance
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    precision = precision_score(y_test, yhat)
    recall = recall_score(y_test, yhat)
    print(f'Metrics for {algo}, Accuracy - {accuracy}, Recall - {recall}, Precision - {precision}')
    
# Save the best model
with open('gradientboosted.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)
    
# Make prediction off of test.csv using GB model
df_test = pd.read_csv('test.csv')
test_abt = preprocessing(df_test.copy())

# one hot encode categorical variables
abt_test = pd.get_dummies(test_abt.drop('PassengerId', axis=1))
# Test prediction
yhat_test = fit_models['gb'].predict(abt_test)

# Make kaggle submission ready
submission = pd.DataFrame([df_test['PassengerId'], yhat_test]).T
submission.columns = ['PassengerId', 'Transported']
submission.to_csv('kaggle_submission.csv', index=False)
print(submission.head())