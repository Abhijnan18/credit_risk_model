import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os


def load_data():
    df = pd.read_csv('data/raw/lending_club_loans.csv')
    # Keep relevant columns (adjust based on dataset)
    features = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
                'fico_range_low', 'total_pymnt', 'loan_status']
    df = df[features]
    # Drop rows with missing values
    df = df.dropna()
    # Create binary target (1 = default, 0 = paid)
    df['target'] = df['loan_status'].apply(
        lambda x: 1 if x in ['Charged Off', 'Default'] else 0)
    return df


def preprocess_data():
    # Create directories if they don't exist
    os.makedirs('data/processed/train_test_data', exist_ok=True)

    df = load_data()

    # Save cleaned data
    df.to_csv('data/processed/cleaned_data.csv', index=False)

    # Split data
    X = df.drop(['target', 'loan_status'], axis=1)
    y = df['target']

    # Handle categorical features
    X['term'] = X['term'].str.extract('(\d+)').astype(int)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Save original splits
    X_train.to_csv(
        'data/processed/train_test_data/X_train_original.csv', index=False)
    X_test.to_csv('data/processed/train_test_data/X_test.csv', index=False)
    y_train.to_csv(
        'data/processed/train_test_data/y_train_original.csv', index=False)
    y_test.to_csv('data/processed/train_test_data/y_test.csv', index=False)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Save resampled data
    X_train_res.to_csv(
        'data/processed/train_test_data/X_train.csv', index=False)
    y_train_res.to_csv(
        'data/processed/train_test_data/y_train.csv', index=False)

    print("All data files processed and saved!")


if __name__ == "__main__":
    preprocess_data()
