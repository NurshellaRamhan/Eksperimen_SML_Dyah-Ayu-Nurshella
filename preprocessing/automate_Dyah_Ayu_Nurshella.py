import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocessing_pipeline(
    input_path,
    target_column,
    output_train_path,
    output_test_path
):
    """
    Pipeline preprocessing otomatis:
    - Load dataset
    - Encoding target
    - Split data
    - Scaling
    - Simpan hasil preprocessing
    """

    # Buat folder output
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

    # 1. Load dataset (Wine Quality pakai separator ;)
    df = pd.read_csv(input_path, sep=";")

    # 2. Encoding target (quality)
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    # 3. Pisahkan fitur & target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Simpan ke CSV
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df[target_column] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df[target_column] = y_test.values

    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)

    print("Preprocessing Wine Quality selesai")
    print("Train:", output_train_path)
    print("Test :", output_test_path)


if __name__ == "__main__":
    preprocessing_pipeline(
        input_path="dataset_raw/winequality-red.csv",
        target_column="quality",
        output_train_path="wine_preprocessing/wine_train.csv",
        output_test_path="wine_preprocessing/wine_test.csv"
    )
