import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_excel("train.xlsx")  
test = pd.read_excel("test.xlsx") 

x=train.drop(columns=['target'])
y=train['target']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define preprocessing steps
numeric_features = X_train.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define XGBoost classifier
xgb_classifier = XGBClassifier()

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_classifier)
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on test data
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
y_pred=label_encoder.inverse_transform(y_pred)

pred=pipeline.predict(test)
new_df=label_encoder.inverse_transform(pred)

print(y_pred)
# Evaluate model
print(f"XGBoost Accuracy: {accuracy:.2f}")
