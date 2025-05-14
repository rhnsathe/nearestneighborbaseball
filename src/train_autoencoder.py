import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("player_season_proxies.csv")
z_cols = [col for col in df.columns if col.endswith("_z")]
print(f"Using {len(z_cols)} features:", z_cols)

missing_counts = df[z_cols].isnull().sum(axis=1)
mask_keep = missing_counts <= 2
dropped = (~mask_keep).sum()
kept = mask_keep.sum()
total = len(df)

df_clean = df[mask_keep].copy()

df_clean[z_cols] = df_clean[z_cols].fillna(0.0)
X_raw = df_clean[z_cols].values.astype("float32")

# ensure no NaNs or Infs
print("Post‑impute NaNs:", np.isnan(X_raw).sum(),
      "Post‑impute Infs:", np.isinf(X_raw).sum())

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

dim_input = X.shape[1]   
dim_latent = 3     

inp = Input(shape=(dim_input,), name="encoder_input")
e1 = Dense(64, activation="relu", name="enc_dense1")(inp)
e2 = Dense(32, activation="relu", name="enc_dense2")(e1)
lat = Dense(dim_latent, activation=None, name="latent")(e2)

d1 = Dense(32, activation="relu", name="dec_dense1")(lat)
d2 = Dense(64, activation="relu", name="dec_dense2")(d1)
out = Dense(dim_input, activation=None, name="reconstruction")(d2)

autoencoder = Model(inputs=inp, outputs=out, name="autoencoder")
encoder = Model(inputs=inp, outputs=lat, name="encoder")
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()
es = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
autoencoder.fit(
    X, X,
    validation_split=0.2,
    epochs=100,
    batch_size=256,
    callbacks=[es],
    verbose=2
)

Z = encoder.predict(X)
latent_df = pd.DataFrame(Z, columns=[f"e{i+1}" for i in range(dim_latent)])

for col in ["playerID", "yearID", "nameFirst", "nameLast"]:
    latent_df[col] = df_clean[col].values

latent_df.to_csv("player_season_latent.csv", index=False)
print(f"Wrote {len(latent_df)} rows to player_season_latent.csv")