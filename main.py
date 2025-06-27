# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import gc
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
training_data = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
additional_data = pd.read_csv('/kaggle/input/d/irakozekelly/fertilizer-prediction/Fertilizer Prediction.csv')

# Display column information and data preview
additional_data.columns, training_data.columns
training_data.head()
test_data.head()
test_data.shape, training_data.shape

# Remove ID columns as they are not needed for prediction
training_data = training_data.drop(columns=['id'])
test_data = test_data.drop(columns=['id'])

# Combine training data with additional data to increase dataset size
training_data = pd.concat([training_data, additional_data], ignore_index=True)

# Display information about the training dataset
training_data.info()

# Analyze categorical columns in training data
categorical_features_train = training_data.select_dtypes(include=['object']).columns
unique_values_train = {col: training_data[col].nunique() for col in categorical_features_train}
for col, unique_count in unique_values_train.items():
    print(f"{col}: {unique_count} unique values")
    
gc.collect()  # Garbage collection to free memory

# Analyze categorical columns in test data
categorical_features_test = test_data.select_dtypes(include=['object']).columns
unique_values_test = {col: test_data[col].nunique() for col in categorical_features_test}
for col, unique_count in unique_values_test.items():
    print(f"{col}: {unique_count} unique values")
    
gc.collect()

training_data.columns

import seaborn as sns

# Calculate and display missing value percentages for training and test data
missing_values_train = training_data.isna().mean() * 100
missing_values_test = test_data.isna().mean() * 100

print("Columns in training data with more than 10% missing values:")
print(missing_values_train[missing_values_train > 0])

print("\nColumns in test data with more than 10% missing values:")
print(missing_values_test[missing_values_test > 0])

# Analyze missing values in training data
missing_value_counts = training_data.isnull().sum()
missing_value_counts = missing_value_counts[missing_value_counts > 0]

# Visualize missing values in training data using a bar plot
if not missing_value_counts.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_value_counts.index, y=missing_value_counts.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Missing Values')
    plt.title('Missing Values per Feature')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

# Define threshold for removing columns with high missing values
missing_value_threshold = 0.95

# Remove columns with high percentage of missing values
high_missing_columns = training_data.columns[training_data.isnull().mean() > missing_value_threshold]
training_data = training_data.drop(columns=high_missing_columns)
test_data = test_data.drop(columns=high_missing_columns)

# Handle missing values in both training and test data
for column in training_data.columns:
    if training_data[column].isnull().any():      
        if training_data[column].dtype == 'object':
            # For categorical columns, fill with mode
            mode_value = training_data[column].mode()[0]
            training_data[column].fillna(mode_value, inplace=True)
            test_data[column].fillna(mode_value, inplace=True)     
        else:
            # For numerical columns, fill with median
            median_value = training_data[column].median()
            training_data[column].fillna(median_value, inplace=True)
            test_data[column].fillna(median_value, inplace=True)

from dython.nominal import associations

# Visualize correlation matrix for categorical features
associations_matrix = associations(training_data[:10000], nominal_columns='all', plot=False)
correlation_matrix = associations_matrix['corr']
plt.figure(figsize=(20, 8))
plt.gcf().set_facecolor('#FFFDD0') 
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix including Categorical Features')
plt.show()

# Get categorical columns except target variable
categorical_columns = training_data.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns[categorical_columns != 'Fertilizer Name']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Apply ordinal encoding to categorical features
training_data[categorical_columns] = ordinal_encoder.fit_transform(training_data[categorical_columns].astype(str))
test_data[categorical_columns] = ordinal_encoder.transform(test_data[categorical_columns].astype(str))

# Encode target variable
label_encoder = LabelEncoder()
training_data['Fertilizer Name'] = label_encoder.fit_transform(training_data['Fertilizer Name'])

# Separate features and target
target = training_data['Fertilizer Name'] 
features = training_data.drop(['Fertilizer Name'], axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Define XGBoost classifier hyperparameters
xgb_params = {
    'objective': 'multi:softprob',  
    'num_class': len(np.unique(target)), 
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'max_bin': 128,
    'colsample_bytree': 0.3, 
    'colsample_bylevel': 1,  
    'colsample_bynode': 1,  
    'tree_method': 'hist',  
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'device': "cuda",
    'enable_categorical': True,
    'n_estimators': 10000,
    'early_stopping_rounds': 50,
}

# Initialize and train XGBoost model
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_train)),
    n_estimators=3200,
    learning_rate=0.045,         
    max_depth=7,                
    colsample_bytree=0.6,       
    colsample_bylevel=0.8,      
    subsample=0.8,
)

# Train the model
xgb_model.fit(X_train, y_train)

# Generate probability predictions for validation set
validation_probabilities = xgb_model.predict_proba(X_val)
top_3_predictions = np.argsort(validation_probabilities, axis=1)[:, -3:][:, ::-1]  
actual_labels = [[label] for label in y_val]

# Define MAP@3 evaluation metric
def calculate_map3(actual, predicted, k=3):
    def average_precision_at_k(actual_val, predicted_val, k):
        predicted_val = predicted_val[:k]
        score = 0.0
        num_hits = 0
        seen_predictions = set()
        for i, pred in enumerate(predicted_val):
            if pred in actual_val and pred not in seen_predictions:
                num_hits += 1
                score += num_hits / (i + 1.0)
                seen_predictions.add(pred)
        return score / min(len(actual_val), k)
    return np.mean([average_precision_at_k(a, p, k) for a, p in zip(actual, predicted)])

# Calculate and display MAP@3 score
map3_score = calculate_map3(actual_labels, top_3_predictions)
print(f"✅ MAP@3 Score: {map3_score:.5f}")

# Initialize LIME explainer for model interpretability
import lime
import lime.lime_tabular

prediction_function = lambda x: xgb_model.predict_proba(x).astype(float)
feature_matrix = X_train.values
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    feature_matrix,
    feature_names=X_train.columns,
    kernel_width=5
)

# Generate and display LIME explanation for a sample instance
sample_instance = X_val.iloc[[15066]].values[0]
explanation = lime_explainer.explain_instance(sample_instance, prediction_function, num_features=15)
explanation.show_in_notebook(show_all=False)

# Generate predictions for test set and create submission file
test_probabilities = xgb_model.predict_proba(test_data)
top_3_test_predictions = np.argsort(test_probabilities, axis=1)[:, -3:][:, ::-1]
top_3_fertilizer_names = label_encoder.inverse_transform(top_3_test_predictions.ravel()).reshape(top_3_test_predictions.shape)
submission_df = pd.DataFrame({
    'id': submission_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_fertilizer_names]
})
submission_df.to_csv('submission.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")