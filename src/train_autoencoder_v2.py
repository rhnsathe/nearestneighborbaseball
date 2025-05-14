import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("player_season_proxies_v2.csv")
z_cols = [c for c in df.columns if c.endswith("_z")]
print("Training on features:", z_cols)
df_clean = df.dropna(subset=z_cols).reset_index(drop=True)
X_raw    = df_clean[z_cols].values.astype("float32")
scaler = StandardScaler()
X      = scaler.fit_transform(X_raw)
dim_input  = X.shape[1]
dim_latent = min(3, dim_input // 2)

inp = Input(shape=(dim_input,), name="input")
e1  = Dense(64, activation="relu")(inp)
lat = Dense(dim_latent, activation=None, name="latent")(e1)
d1  = Dense(64, activation="relu")(lat)
out = Dense(dim_input, activation=None)(d1)

autoencoder = Model(inputs=inp, outputs=out, name="autoencoder_v2")
encoder     = Model(inputs=inp, outputs=lat, name="encoder_v2")
autoencoder.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
autoencoder.fit(
    X, X,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=[es],
    verbose=2
)

Z = encoder.predict(X)
latent_df = pd.DataFrame(Z, columns=[f"e{i+1}" for i in range(Z.shape[1])])
latent_df[["player_id","year"]] = df_clean[["player_id","year"]]
latent_df.to_csv("player_season_latent_v2.csv", index=False)
print(f"Wrote {len(latent_df)} rows to player_season_latent_v2.csv")