# ============================================================
# TEST SCRIPT – TENYI VO PIG WEIGHT MODEL 
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


# ===================== PATHS =====================
MODEL_PATH = "paper_outputs/best_model.keras"
IMG_DIR = "/media/dell/My Book2/velato Nyekha data/Velato Data"
CSV_PATH = "/media/dell/My Book2/velato Nyekha data/MORPHOMETRIC.csv"

IMG_SIZE = 224
BATCH_SIZE = 16


# ====================================================
# 1. LOAD MODEL
# ====================================================
print("Loading trained model...")
model = load_model(MODEL_PATH)


# ====================================================
# 2. LOAD CSV (ROBUST COLUMN HANDLING)
# ====================================================
print("Reading CSV...")
df = pd.read_csv(CSV_PATH)

# clean column names safely
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(r"[^\w]", "", regex=True)
)

print("Detected columns:", list(df.columns))

# auto-detect animal id column
possible_cols = ["animal_no","animal","animalid","id","tag"]

found = None
for col in possible_cols:
    if col in df.columns:
        found = col
        break

if found is None:
    raise ValueError(f"No animal ID column found. Columns present: {list(df.columns)}")

df["animal_no"] = df[found].astype(str).str.strip().str.lower()

# encode sex if present
if "sex" in df.columns:
    df["sex"] = LabelEncoder().fit_transform(df["sex"].astype(str))
else:
    df["sex"] = 0


# ====================================================
# 3. LOAD IMAGES
# ====================================================
images, labels = [], []

print("Loading images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    folder = os.path.join(IMG_DIR, row["animal_no"])
    if not os.path.isdir(folder):
        continue

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg",".jpeg",".png")):
            img = cv2.imread(os.path.join(folder,file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            images.append(img.astype(np.float32)/255.0)
            labels.append(row.to_dict())

X_img = np.array(images, dtype=np.float32)
tab_df = pd.DataFrame(labels)
y_true = tab_df["weight"].astype(float).values

print("Total test images:", len(X_img))


# ====================================================
# 4. TABULAR FEATURES (SAME AS TRAINING)
# ====================================================
features = [
"sex","age_in_months","face_length","height_at_wither",
"heart_girth","middle_girth","body_length","neck_length",
"face_width","ear_length","shoulder_width","hip_width",
"tail_length","punch_girth"
]

features = [f for f in features if f in tab_df.columns]

poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
X_tab = poly.fit_transform(tab_df[features].fillna(0))

scaler = StandardScaler()
X_tab = scaler.fit_transform(X_tab)


# ====================================================
# 5. EXTRACT IMAGE FEATURES
# ====================================================
print("Extracting image features...")

resnet = ResNet50(include_top=False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3),
                  pooling="avg")

def extract(model, imgs):
    feats=[]
    for i in tqdm(range(0,len(imgs),BATCH_SIZE)):
        feats.append(model.predict(imgs[i:i+BATCH_SIZE],verbose=0))
    return np.vstack(feats)

X_img_f = extract(resnet,X_img)


# ====================================================
# 6. FUSION
# ====================================================
X_test = np.hstack([X_img_f,X_tab])


# ====================================================
# 7. PREDICTION
# ====================================================
print("Evaluating model...")
y_pred = model.predict(X_test).flatten()


# ====================================================
# 8. METRICS (ACCURACY EQUIVALENT)
# ====================================================
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)

print("\n========== TEST PERFORMANCE ==========")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}")
print("======================================")
