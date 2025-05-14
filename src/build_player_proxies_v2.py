import pandas as pd
import numpy as np

ADV_STATS_CSV       = "../statcast/advancedstats.csv"
PLAYERS_MINIMAL_CSV = "players_minimal.csv"

adv = pd.read_csv(ADV_STATS_CSV, low_memory=False)
adv = adv[adv["year"] >= 2015].copy()

name_col = [c for c in adv.columns if ',' in c and 'last_name' in c][0]
adv.drop(columns=[name_col], inplace=True)

KEYS      = ["player_id","year"]
feat_cols = [c for c in adv.columns if c not in KEYS]

adv["decade"] = (adv["year"] // 10) * 10

for col in feat_cols:
    grp   = adv.groupby("decade")[col]
    mu    = grp.transform("mean")
    sigma = grp.transform("std").replace(0, np.nan)
    adv[f"{col}_z"] = (adv[col] - mu) / sigma

players = pd.read_csv(PLAYERS_MINIMAL_CSV, low_memory=False)
proxies = adv.merge(players, on="player_id", how="left")

out_cols = ["player_id","nameFirst","nameLast","year"] \
         + [f"{c}_z" for c in feat_cols]

proxies[out_cols].to_csv("player_season_proxies_v2.csv", index=False)
print(f"Wrote {len(proxies)} rows to player_season_proxies_v2.csv")