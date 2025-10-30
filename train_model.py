import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Load dataset ---
df = pd.read_csv("amazon_delivery.csv")

# --- Drop irrelevant columns ---
df = df.drop(["Order_ID"], axis=1)

# --- Handle missing values ---
df = df.dropna()

# --- Encode categorical columns ---
label_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Convert time columns ---
df["Order_Timestamp"] = pd.to_datetime(df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str), errors="coerce")
df["Pickup_Timestamp"] = pd.to_datetime(df["Order_Date"].astype(str) + " " + df["Pickup_Time"].astype(str), errors="coerce")

# --- Compute preparation time (minutes) ---
df["Prep_Time"] = (df["Pickup_Timestamp"] - df["Order_Timestamp"]).dt.total_seconds() / 60

# --- Compute distance (rough estimate using lat/long differences) ---
df["Distance_km"] = ((df["Drop_Latitude"] - df["Store_Latitude"])**2 + (df["Drop_Longitude"] - df["Store_Longitude"])**2)**0.5 * 111

# --- Drop original date/time columns ---
df = df.drop(["Order_Date", "Order_Time", "Pickup_Time", "Order_Timestamp", "Pickup_Timestamp"], axis=1)

# --- Define target and features ---
X = df.drop("Delivery_Time", axis=1)
y = df["Delivery_Time"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Save model ---
joblib.dump(model, "delivery_time_model.pkl")
print("✅ Model trained and saved successfully!")
print(f"Features used: {list(X.columns)}")
