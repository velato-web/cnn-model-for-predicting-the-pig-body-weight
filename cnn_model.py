# ============================================================
# TENYI VO PIG BODY WEIGHT ESTIMATION
# ============================================================

# ==================== ENVIRONMENT ====================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==================== MATPLOTLIB BACKEND FIX ====================
import matplotlib
matplotlib.use("Agg")  # prevents Qt crash

# ==================== IMPORTS ====================
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# ==================== PARAMETERS ====================
IMG_DIR = "/media/dell/My Book2/velato Nyekha data/Velato Data"
CSV_PATH = "/media/dell/My Book2/velato Nyekha data/MORPHOMETRIC.csv"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 120
LAST_CONV_LAYER = "conv5_block3_out"

OUT_DIR = "paper_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ====================================================
# 1. LOAD CSV
# ====================================================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

if "animal_no" not in df.columns:
    df["animal_no"] = df.iloc[:,0].astype(str)

df["animal_no"] = df["animal_no"].astype(str).str.strip().str.lower()
df = df[~df["animal_no"].isin(["no tag","notag","unknown","na","nan",""])]

if "sex" in df.columns:
    df["sex"] = LabelEncoder().fit_transform(df["sex"].astype(str))
else:
    df["sex"] = 0

# ====================================================
# 2. LOAD IMAGES
# ====================================================
images, labels = [], []
skipped = set()

print("📥 Loading images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    folder = os.path.join(IMG_DIR, row["animal_no"])
    if not os.path.isdir(folder):
        skipped.add(row["animal_no"])
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
y = tab_df["weight"].astype(float).values

print("Images:",len(X_img))
print("Skipped:",skipped)

# ====================================================
# 3. TABULAR FEATURES
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
X_tab = StandardScaler().fit_transform(X_tab)

# ====================================================
# 4. SPLIT
# ====================================================
X_img_temp, X_img_test, X_tab_temp, X_tab_test, y_temp, y_test = train_test_split(
    X_img, X_tab, y, test_size=0.15, random_state=42
)

X_img_train, X_img_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
    X_img_temp, X_tab_temp, y_temp, test_size=0.176, random_state=42
)

# ====================================================
# 5. RESNET FEATURE EXTRACTOR
# ====================================================
resnet = ResNet50(include_top=False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3),
                  pooling="avg")

for layer in resnet.layers[:-20]:
    layer.trainable = False

feature_extractor = Model(resnet.input,resnet.output)

def extract_features(model,imgs):
    feats=[]
    for i in tqdm(range(0,len(imgs),BATCH_SIZE)):
        feats.append(model.predict(imgs[i:i+BATCH_SIZE],verbose=0))
    return np.vstack(feats)

X_img_train_f = extract_features(feature_extractor,X_img_train)
X_img_val_f   = extract_features(feature_extractor,X_img_val)
X_img_test_f  = extract_features(feature_extractor,X_img_test)

# ====================================================
# 6. FEATURE FUSION
# ====================================================
X_train = np.hstack([X_img_train_f,X_tab_train])
X_val   = np.hstack([X_img_val_f,X_tab_val])
X_test  = np.hstack([X_img_test_f,X_tab_test])

# ====================================================
# 7. REGRESSION MODEL
# ====================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256,activation="relu"),
    Dropout(0.3),
    Dense(128,activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam",loss="mse",metrics=["mae"])

history = model.fit(
    X_train,y_train,
    validation_data=(X_val,y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=12,restore_best_weights=True),
        ModelCheckpoint(os.path.join(OUT_DIR,"best_model.keras"),
                        save_best_only=True)
    ],
    verbose=1
)

# ====================================================
# 8. EVALUATION
# ====================================================
def evaluate(y_true,y_pred):
    return (
        np.sqrt(mean_squared_error(y_true,y_pred)),
        mean_absolute_error(y_true,y_pred),
        mean_absolute_percentage_error(y_true,y_pred),
        r2_score(y_true,y_pred)
    )

y_train_pred = model.predict(X_train).flatten()
y_val_pred   = model.predict(X_val).flatten()
y_test_pred  = model.predict(X_test).flatten()

metrics_df = pd.DataFrame([
["Train",*evaluate(y_train,y_train_pred)],
["Validation",*evaluate(y_val,y_val_pred)],
["Test",*evaluate(y_test,y_test_pred)]
],columns=["Set","RMSE","MAE","MAPE","R2"])

metrics_df.to_csv(os.path.join(OUT_DIR,"Table_1_Performance.csv"),index=False)
print(metrics_df)

# ====================================================
# FIGURE 1: ACTUAL vs PREDICTED
# ====================================================
plt.figure(figsize=(6,6))
plt.scatter(y_test,y_test_pred,alpha=0.7,edgecolor="k")
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],"r--")
plt.xlabel("Actual Weight (kg)")
plt.ylabel("Predicted Weight (kg)")
plt.title("Actual vs Predicted (Test)")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"Figure_1.png"),dpi=300)
plt.close()

# ====================================================
# FIGURE 2: TRAINING CURVES
# ====================================================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["mae"])
plt.plot(history.history["val_mae"])
plt.title("MAE"); plt.grid()

plt.subplot(1,2,2)
plt.plot(np.sqrt(history.history["loss"]))
plt.plot(np.sqrt(history.history["val_loss"]))
plt.title("RMSE"); plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"Figure_2.png"),dpi=300)
plt.close()

# ====================================================
# 9. GRAD-CAM
# ====================================================
def gradcam(img_array,model,layer):
    grad_model = Model(model.input,
        [model.get_layer(layer).output,model.output])

    with tf.GradientTape() as tape:
        conv,pred = grad_model(img_array)
        loss = pred[:,0]

    grads = tape.gradient(loss,conv)
    pooled = tf.reduce_mean(grads,axis=(0,1,2))
    heatmap = tf.reduce_sum(conv[0]*pooled,axis=-1)
    heatmap = np.maximum(heatmap,0)
    return heatmap/(np.max(heatmap)+1e-8)

# ====================================================
# 10. GRADCAM FIGURES
# ====================================================
def save_gradcams(dataset,name):
    idxs = np.linspace(0,len(dataset)-1,3,dtype=int)
    for i,idx in enumerate(idxs):
        img = dataset[idx]
        heat = gradcam(np.expand_dims(img,0),
                       feature_extractor,
                       LAST_CONV_LAYER)
        heat = cv2.resize(heat,(IMG_SIZE,IMG_SIZE))
        overlay = cm.jet(heat)[:,:,:3]*0.4 + img

        plt.figure(figsize=(12,4))
        for j,(im,title) in enumerate(zip(
            [img,heat,overlay],
            ["Original","Heatmap","Overlay"])):
            plt.subplot(1,3,j+1)
            plt.imshow(im,cmap="jet" if j==1 else None)
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR,f"{name}_{i+1}.png"),dpi=300)
        plt.close()

save_gradcams(X_img_val,"Figure_3_Validation")
save_gradcams(X_img_test,"Figure_4_Test")
