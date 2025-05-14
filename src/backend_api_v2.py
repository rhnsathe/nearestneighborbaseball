import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

DF = (
    pd.read_csv("player_season_latent_v2.csv")
      .dropna(subset=["e1","e2","e3"])
      .reset_index(drop=True)
)
meta = DF[["player_id","year"]].to_dict(orient="index")
PLAYERS_CSV = "../statcast/players_minimal.csv"
players_df = pd.read_csv(
    PLAYERS_CSV,
    usecols=["player_id", "nameFirst", "nameLast"],
    dtype={"player_id": int, "nameFirst": str, "nameLast": str},
    low_memory=False,
    encoding="latin1"
)

players_df["fullName"] = (
    players_df["nameFirst"].str.strip()
    + " "
    + players_df["nameLast"].str.strip()
)

players_map = players_df.set_index("player_id")[["nameFirst","nameLast"]].to_dict("index")

features = DF[["e1","e2","e3"]].values.astype("float32")
d        = features.shape[1]
index    = faiss.IndexFlatL2(d)
index.add(features)

app = FastAPI(title="Player Similarity API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarResult(BaseModel):
    player_id: int
    nameFirst: str
    nameLast:  str
    year:      int
    distance:  float

class SimilarResponse(BaseModel):
    query_player: int
    query_year:   int
    results:      List[SimilarResult]

class PlayerSuggestion(BaseModel):
    player_id: int
    nameFirst: str
    nameLast:  str

@app.get("/similar_v2/{player_id}", response_model=SimilarResponse)
def similar_v2(
    player_id: int,
    year: Optional[int] = Query(None, description="Restrict to this season"),
    k:   int = Query(5, ge=1, le=20, description="Neighbors to return")
):
    subset = DF[(DF.player_id==player_id) & ((DF.year==year) if year else True)]
    if subset.empty:
        raise HTTPException(404, f"No data for {player_id}{' in '+str(year) if year else ''}")
    qidx = subset.index.max()
    qvec = features[qidx:qidx+1]

    D, I = index.search(qvec, k+1)
    out = []
    for dist, idx in zip(D[0], I[0]):
        if idx == qidx:
            continue
        pid = int(meta[idx]["player_id"])
        yr  = int(meta[idx]["year"])
        nm = players_map.get(pid, {"nameFirst": "", "nameLast": ""})
        out.append(SimilarResult(
            player_id=pid,
            nameFirst=nm["nameFirst"],
            nameLast=nm["nameLast"],
            year=yr,
            distance=float(dist)
        ))
        if len(out) >= k:
            break

    return SimilarResponse(
        query_player=int(player_id),
        query_year=int(DF.at[qidx,"year"]),
        results=out
    )

@app.get("/players_v2", response_model=List[PlayerSuggestion])
def players(q: str = Query(..., min_length=2, description="Substring to search")):
    mask    = players_df["fullName"].str.contains(q, case=False, na=False)
    matches = players_df[mask].head(10)

    return [
        PlayerSuggestion(
            player_id=int(row.player_id),
            nameFirst=row.nameFirst,
            nameLast=row.nameLast
        )
        for _, row in matches.iterrows()
    ]