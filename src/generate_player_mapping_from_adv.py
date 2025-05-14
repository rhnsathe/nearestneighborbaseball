import pandas as pd

ADV_STATS_CSV = "../statcast/advancedstats.csv"
adv = pd.read_csv(ADV_STATS_CSV, low_memory=False)
name_col = [c for c in adv.columns if ',' in c and 'last_name' in c][0]

mapping = (
    adv[["player_id", name_col]]
      .drop_duplicates(subset=["player_id"])
      .reset_index(drop=True)
)

mapping[["nameLast","nameFirst"]] = (
    mapping[name_col]
      .str
      .split(", ", n=1, expand=True)
)

players_minimal = mapping[["player_id","nameFirst","nameLast"]]
players_minimal.to_csv("players_minimal.csv", index=False)
print(f"Wrote {len(players_minimal)} players to players_minimal.csv")