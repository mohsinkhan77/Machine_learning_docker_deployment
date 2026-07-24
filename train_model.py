
#PROJECT 6 - PART 1: TRAIN AND SAVE THE MODEL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # used to save/load Python objects (our trained model) to/from disk

np.random.seed(42)


# STEP 1: GENERATE AND SAVE A SYNTHETIC DATASET AS CSV

n = 2000
df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "time_on_site_minutes": np.random.exponential(5, n).clip(0, 60),
    "pages_viewed": np.random.poisson(4, n).clip(1, 30),
    "past_purchases": np.random.poisson(1.5, n).clip(0, 20),
    "cart_value": np.random.exponential(50, n).clip(0, 500),
})

purchase_score = (
    (df["time_on_site_minutes"] / 60) * 2 +
    (df["pages_viewed"] / 30) * 1.5 +
    (df["past_purchases"] / 20) * 2 +
    (df["cart_value"] / 500) * 2.5
)
purchase_score += np.random.normal(0, 0.4, n)
df["will_purchase"] = (purchase_score > purchase_score.median()).astype(int)

df.to_csv("customer_data.csv", index=False)
print(f"Saved customer_data.csv with {len(df)} rows")


# STEP 2: QUICK EDA
df = pd.read_csv("customer_data.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()} total missing cells")
print(f"Purchase rate: {df['will_purchase'].mean():.1%}")
print(f"\nFeature summary:\n{df.describe().round(2)}")


# STEP 3: TRAIN THE MODEL

feature_cols = ["age", "time_on_site_minutes", "pages_viewed", "past_purchases", "cart_value"]
X = df[feature_cols]
y = df["will_purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)


# STEP 4: EVALUATE

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {acc:.3f}")
print(classification_report(y_test, preds, target_names=["No Purchase", "Purchase"]))


# STEP 5: SAVE THE TRAINED MODEL TO DISK

joblib.dump(model, "purchase_model.pkl")
print("\nSaved trained model as purchase_model.pkl")

joblib.dump(feature_cols, "feature_columns.pkl")
print("Saved feature_columns.pkl")
