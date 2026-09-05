"""Common preprocessing utilities."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_features_target(df, target_column):
    """Separate input features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Create a reproducible train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def fit_standard_scaler(X_train):
    """Fit a StandardScaler using training data only."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled
