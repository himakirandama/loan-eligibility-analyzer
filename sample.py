import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import joblib
warnings.filterwarnings("ignore")


def load_data():

    df = pd.read_csv("loan_approval_dataset.csv")
    print("First 5 rows before cleaning:")
    print(df.head())
    print("\nMissing values per column (before cleaning):")
    print(df.isnull().sum())


    df.columns = df.columns.str.strip()
    

    df['loan_status'] = df['loan_status'].str.strip()
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

    print("\nColumn names:")
    print(df.columns.to_list())

    print("\nFirst 5 rows after cleaning:")
    print(df.head())

    return df

def one_hot_encode(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    print("\nColumns after one-hot encoding:")
    df.rename(columns={
    "education_ Graduate": "education_graduate",
    "self_employed_ No": "self_employed_no"
        }, inplace=True)   
    print(df.columns.to_list())
    
    return df

def correlation_analysis(df, threshold=1):
    df = df.drop('loan_id', axis=1)  
    n = df.drop(columns=['loan_status','loan_amount'], axis=1) 
    corr_matrix = n.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.show()

    col_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_corr.add(corr_matrix.columns[i])
    
    print("\nHighly correlated features removed (> {}): {}".format(threshold, col_corr))

    df = df.drop(columns=col_corr)
    print("\nColumns after dropping correlated features:")
    print(df.columns.to_list())

    return df, col_corr

def mutual_information(df,threshold=0.8, top_k=10):
    X = df.drop(columns='loan_status', axis=1)
    y = df['loan_status']
    mi = mutual_info_classif(X, y, n_neighbors=3, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    print("\nMutual Information Scores :\n")
    print(mi_series)

    if threshold is not None:
        selected_features = mi_series[mi_series >= threshold].index.tolist()
        print(f"\nFeatures with MI ≥ {threshold}:\n", selected_features)
    else:
        top_features = mi_series.head(top_k).index.tolist()
        print(f"\nTop {top_k} Features:\n", top_features)

    
    top_plot = mi_series.head(top_k)
    plt.figure(figsize=(12, 6))
    plt.bar(top_plot.index, top_plot.values)
    plt.xticks(rotation=90)
    plt.title('Mutual Information Scores vs Features')
    plt.xlabel('Feature')
    plt.ylabel('Mutual Information Score')
    plt.tight_layout()
    plt.show()

    return X,y,mi_series


def xgb_classifier(X, y, mi_series, top_k=10):
    # Select top_k features based on MI ranking
    selected_features = mi_series.index[:top_k]
    X_selected = X[selected_features]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\n=== XGBoost with Top {top_k} Features ===")
    print("Selected Features:", list(selected_features))
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print("\nClassification Report (Test Data):\n", classification_report(y_test, y_test_pred))
    print("Confusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred))

    joblib.dump(model, "loan_approval_model.pkl")
    print("\nModel saved as 'loan_approval_model.pkl'")
    print(model.get_booster().feature_names)
    return model, selected_features



if __name__ == "__main__":
    df = load_data()
    df = one_hot_encode(df)
    df,col_corr = correlation_analysis(df, threshold=0.8)
    X, y, mi_series = mutual_information(df, top_k=10)
    best_model, selected_features = xgb_classifier(X, y, mi_series, top_k=10)