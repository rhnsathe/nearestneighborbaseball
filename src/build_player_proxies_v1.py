import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

PLAYERS_CSV   = "../lahman/People.csv"
BATTING_CSV   = "../lahman/Batting.csv"
FIELDING_CSV  = "../lahman/Fielding.csv"
TEAMS_CSV     = "../lahman/Teams.csv"

teams = pd.read_csv(TEAMS_CSV, low_memory=False)
teams = teams.rename(columns={"2B":"doubles","3B":"triples","R":"runs_scored"})
for c in ["BB","HBP","SF","SO","doubles","triples","HR","H","AB"]:
    teams[c] = teams[c].fillna(0)
teams["singles"] = teams["H"] - teams["doubles"] - teams["triples"] - teams["HR"]
mlb = teams[teams["lgID"].isin(["AL","NL"])]

yearly_weights = {}
for year, grp in mlb.groupby("yearID"):
    if len(grp) < 20: continue
    X   = grp[["BB","HBP","singles","doubles","triples","HR"]]
    Xc  = sm.add_constant(X)
    y   = grp["runs_scored"]
    m   = sm.OLS(y, Xc).fit()
    p   = m.params.to_dict()
    yearly_weights[year] = {
        "w_BB":  p.get("BB",  np.nan),
        "w_HBP": p.get("HBP", np.nan),
        "w_1B":  p.get("singles", np.nan),
        "w_2B":  p.get("doubles", np.nan),
        "w_3B":  p.get("triples", np.nan),
        "w_HR":  p.get("HR", np.nan),
    }

weights_df = (
    pd.DataFrame.from_dict(yearly_weights, orient="index")
      .rename_axis("yearID")
      .reset_index()
)
weights_df.to_csv("yearly_linear_weights.csv", index=False)
print(f"Wrote yearly weights for {len(weights_df)} seasons.")

players = pd.read_csv(PLAYERS_CSV, encoding="latin-1", low_memory=False)

bat = pd.read_csv(BATTING_CSV, low_memory=False)
if "G_batting" in bat.columns:
    bat.drop(columns=["G_batting"], inplace=True)

bat.rename(columns={"G": "G_batting"}, inplace=True)

for c in ["BB","HBP","SF","SH","IBB","SO","2B","3B","HR","H","AB","SB","CS","R","G","G_batting","RBI"]:
    if c in bat.columns:
        bat[c] = bat[c].fillna(0)
bat = bat.rename(columns={"2B":"doubles","3B":"triples","G":"G_batting"})
bat["singles"] = bat["H"] - bat["doubles"] - bat["triples"] - bat["HR"]

bat["AVG"]  = bat["H"] / bat["AB"].replace(0, np.nan)
bat["OBP"]  = (bat["H"] + bat["BB"] + bat["HBP"]) / (bat["AB"] + bat["BB"] + bat["HBP"] + bat["SF"])
bat["SLG"]  = (bat["singles"] + 2*bat["doubles"] + 3*bat["triples"] + 4*bat["HR"]) / bat["AB"].replace(0, np.nan)
bat["ISO"]  = bat["SLG"] - bat["AVG"]
den_babip = bat["AB"] - bat["SO"] - bat["HR"] + bat["SF"]
bat["BABIP"] = (bat["H"] - bat["HR"]) / den_babip.replace(0, np.nan)

bat = bat.merge(weights_df, on="yearID", how="left")
num = (
      bat["w_BB"]  * bat["BB"]
    + bat["w_HBP"] * bat["HBP"]
    + bat["w_1B"]  * bat["singles"]
    + bat["w_2B"]  * bat["doubles"]
    + bat["w_3B"]  * bat["triples"]
    + bat["w_HR"]  * bat["HR"]
)
den_woba = bat["AB"] + bat["BB"] - bat["IBB"] + bat["SF"] + bat["HBP"]
bat["wOBA"] = num / den_woba.replace(0, np.nan)

bat["PA"]         = bat["AB"] + bat["BB"] + bat["HBP"] + bat["SH"] + bat["SF"]
bat["OPS"]        = bat["OBP"] + bat["SLG"]
bat["TB"]         = bat["singles"] + 2*bat["doubles"] + 3*bat["triples"] + 4*bat["HR"]
bat["RC"]         = ((bat["H"] + bat["BB"]) * bat["TB"]) / (bat["AB"] + bat["BB"]).replace(0, np.nan)
bat["RC27"]       = bat["RC"] / bat["PA"].replace(0, np.nan) * 27
bat["K_rate"]     = bat["SO"] / bat["PA"].replace(0, np.nan)
bat["BB_rate"]    = bat["BB"] / bat["PA"].replace(0, np.nan)
bat["K_BB"]       = bat["SO"] / bat["BB"].replace(0, np.nan)
bat["R_per_game"] = bat["R"] / bat["G_batting"].replace(0, np.nan)
bat["SB_pct"]     = bat["SB"] / (bat["SB"] + bat["CS"]).replace(0, np.nan)

fld = pd.read_csv(FIELDING_CSV, low_memory=False)
for c in ["PO","A","E","G"]:
    fld[c] = fld[c].fillna(0)
fld_agg = (
    fld.groupby(["playerID","yearID"], as_index=False)
       .agg({"PO":"sum","A":"sum","E":"sum","G":"sum"})
)
fld_agg["fld_pct"]      = (fld_agg["PO"] + fld_agg["A"]) / (fld_agg["PO"] + fld_agg["A"] + fld_agg["E"]).replace(0, np.nan)
fld_agg["range_factor"] = (fld_agg["PO"] + fld_agg["A"]) / fld_agg["G"].replace(0, np.nan)
bat = bat.merge(fld_agg, on=["playerID","yearID"], how="left")

bat["decade"] = (bat["yearID"] // 10) * 10
metrics = [
    "OBP","SLG","ISO","BABIP","wOBA",
    "OPS","RC","RC27",
    "K_rate","BB_rate","K_BB","R_per_game","SB_pct",
    "fld_pct","range_factor"
]
for col in metrics:
    bat[f"{col}_z"] = (
        bat[col]
      - bat.groupby("decade")[col].transform("mean")
    ) / bat.groupby("decade")[col].transform("std")

proxies = bat.merge(players[["playerID","nameFirst","nameLast"]], on="playerID", how="left")
cols = ["playerID","nameFirst","nameLast","yearID"] + [f"{c}_z" for c in metrics]
proxies[cols].to_csv("player_season_proxies.csv", index=False)
print(f"Wrote {len(proxies)} rows to player_season_proxies.csv")
