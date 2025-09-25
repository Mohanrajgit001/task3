# linear_regression_task.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ---------------- SETTINGS ----------------
USE_LOG_TARGET = False  # set True to use log(price)
SAVE_PLOTS_DIR = "plots"
os.makedirs(SAVE_PLOTS_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("housing.csv")
print("Shape:", df.shape)
print(df.head())

print("\nColumns and dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------- CLEANING ----------------
# Drop non-feature cols if present
for c in ["id", "date"]:
    if c in df.columns:
        df = df.drop(columns=c)

# Fill numeric NaNs with median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encode categoricals
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ---------------- FEATURES & TARGET ----------------
if "price" not in df.columns:
    raise SystemExit("Target column 'price' not found. Rename target to 'price'.")

y = df["price"]
X = df.drop(columns=["price"])

# Correlation ranking
corrs = df.corr()["price"].abs().sort_values(ascending=False)
print("\nTop correlations with price:")
print(corrs.head(10))

# Pick top features (excluding target itself)
top_features = corrs.index[1:8].tolist()
X = df[top_features]

# ---------------- TRAIN/TEST SPLIT ----------------
if USE_LOG_TARGET:
    y = np.log1p(y)  # log transform
    print("\nUsing log-transform for target")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------- MULTIPLE LINEAR REGRESSION ----------------
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

if USE_LOG_TARGET:
    # Inverse transform predictions
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMultiple Linear Regression metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 :", r2)

mean_price = y_test.mean()
print(f"MAE % of mean price: {mae/mean_price*100:.2f}%")
print(f"RMSE % of mean price: {rmse/mean_price*100:.2f}%")

# Coefficients
coef_df = pd.DataFrame({"feature": X.columns, "coef_standardized": lr.coef_})
coef_df = coef_df.sort_values(by="coef_standardized", key=abs, ascending=False)
print("\nCoefficients (standardized):")
print(coef_df)
coef_df.to_csv("model_coefficients.csv", index=False)

# Save model
joblib.dump({"model": lr, "scaler": scaler, "features": list(X.columns)}, "linear_regression_model.joblib")
print("\nSaved model to linear_regression_model.joblib")

# ---------------- DIAGNOSTIC PLOTS ----------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.35)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linewidth=2, color="red")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.savefig(f"{SAVE_PLOTS_DIR}/predicted_vs_actual.png")
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(7,4))
plt.hist(residuals, bins=40)
plt.title("Residuals distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{SAVE_PLOTS_DIR}/residuals_hist.png")
plt.close()

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, linestyle="--", color="red")
plt.xlabel("Predicted price")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted")
plt.tight_layout()
plt.savefig(f"{SAVE_PLOTS_DIR}/residuals_vs_predicted.png")
plt.close()

print(f"\nPlots saved in {SAVE_PLOTS_DIR}/")

# ---------------- MULTICOLLINEARITY (VIF) ----------------
def compute_vif(X_df):
    from sklearn.linear_model import LinearRegression
    vif = {}
    for i, col in enumerate(X_df.columns):
        X_rest = X_df.drop(columns=[col])
        r2 = LinearRegression().fit(X_rest, X_df[col]).score(X_rest, X_df[col])
        vif[col] = 1.0 / (1.0 - r2) if (1.0 - r2) != 0 else np.inf
    return pd.Series(vif).sort_values(ascending=False)

print("\nVIF values:")
print(compute_vif(X))

# ---------------- REGULARIZATION ----------------
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5).fit(X_train_s, y_train)
lasso = LassoCV(alphas=None, cv=5, max_iter=5000).fit(X_train_s, y_train)

print("\nRidge R2 (test):", ridge.score(X_test_s, y_test))
print("Lasso R2 (test):", lasso.score(X_test_s, y_test))
print("Lasso kept features:", (lasso.coef_ != 0).sum())

# ---------------- CROSS-VALIDATION ----------------
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
print("\nCross-validation R2 scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())
