import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


# ================= PATHS =================
IMG_DIR = "/media/dell/My Book2/velato Nyekha data/Velato Data"
CSV_PATH = "/media/dell/My Book2/velato Nyekha data/MORPHOMETRIC.csv"

IMG_SIZE = 224
BATCH_SIZE = 16
KFOLDS = 5


# ================= LOAD CSV =================
df = pd.read_csv(CSV_PATH)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace(r"[^\w]", "", regex=True)
)

df["animal_no"] = df["animal_no"].astype(str).str.strip().str.lower()

if "sex" in df.columns:
    df["sex"] = LabelEncoder().fit_transform(df["sex"].astype(str))
else:
    df["sex"] = 0


# ================= LOAD IMAGES =================
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

X_img = np.array(images,dtype=np.float32)
tab_df = pd.DataFrame(labels)
y = tab_df["weight"].astype(float).values


# ================= TABULAR FEATURES =================
features = [
"sex","age_in_months","face_length","height_at_wither",
"heart_girth","middle_girth","body_length","neck_length",
"face_width","ear_length","shoulder_width","hip_width",
"tail_length","punch_girth"
]

features = [f for f in features if f in tab_df.columns]

poly = PolynomialFeatures(2,interaction_only=True,include_bias=False)
X_tab = poly.fit_transform(tab_df[features].fillna(0))
X_tab = StandardScaler().fit_transform(X_tab)


# ================= FEATURE EXTRACTION =================
print("Extracting ResNet features...")

resnet = ResNet50(include_top=False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3),
                  pooling="avg")

def extract(model,imgs):
    feats=[]
    for i in tqdm(range(0,len(imgs),BATCH_SIZE)):
        feats.append(model.predict(imgs[i:i+BATCH_SIZE],verbose=0))
    return np.vstack(feats)

X_img_f = extract(resnet,X_img)

# fuse
X = np.hstack([X_img_f,X_tab])


# ================= K-FOLD VALIDATION =================
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

r2s, rmses, maes = [], [], []

print("\nRunning Cross Validation...")

for fold,(train_idx,test_idx) in enumerate(kf.split(X),1):

    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(256,activation="relu"),
        Dropout(0.3),
        Dense(128,activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam",loss="mse")

    model.fit(X_train,y_train,epochs=60,batch_size=16,verbose=0)

    pred = model.predict(X_test,verbose=0).flatten()

    r2s.append(r2_score(y_test,pred))
    rmses.append(np.sqrt(mean_squared_error(y_test,pred)))
    maes.append(mean_absolute_error(y_test,pred))

    print(f"Fold {fold} R² = {r2s[-1]:.4f}")


# ================= FINAL REPORT =================
print("\n========== CROSS VALIDATION RESULT ==========")
print(f"Mean R²  : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
print(f"Mean RMSE: {np.mean(rmses):.4f}")
print(f"Mean MAE : {np.mean(maes):.4f}")
print("=============================================")
