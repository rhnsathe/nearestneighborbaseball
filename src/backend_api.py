import pandas as pd
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Player Similarity & Lookup API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PLAYERS_CSV = "../lahman/People.csv"
players_df = pd.read_csv(
    PLAYERS_CSV,
    usecols=["playerID", "nameFirst", "nameLast"],
    low_memory=False,
    encoding="latin1",   
    dtype={"playerID": str, "nameFirst": str, "nameLast": str}
)

players_df["fullName"] = (
    players_df["nameFirst"].str.strip()
    + " "
    + players_df["nameLast"].str.strip()
)

EMBEDDINGS_CSV = "player_season_latent.csv"
emb_df = pd.read_csv(
    EMBEDDINGS_CSV,
    dtype={"playerID": str, "yearID": int, "e1": float, "e2": float, "e3": float}
)
emb_df = emb_df.dropna(subset=["e1", "e2", "e3"]).reset_index(drop=True)

meta = emb_df[["playerID", "yearID"]].to_dict(orient="index")

first_names = pd.Series(emb_df['nameFirst'].values, index=emb_df['playerID']).to_dict()
last_names = pd.Series(emb_df['nameLast'].values, index=emb_df['playerID']).to_dict()


features = emb_df[["e1", "e2", "e3"]].values.astype("float32")

d = features.shape[1]
index = faiss.IndexFlatL2(d)
index.add(features)

class SimilarResult(BaseModel):
    playerID: str
    nameFirst: str
    nameLast: str
    yearID: int
    distance: float

class SimilarResponse(BaseModel):
    query_player: str
    query_year: int
    results: List[SimilarResult]

class PlayerSuggestion(BaseModel):
    playerID: str
    nameFirst: str
    nameLast: str

@app.get("/similar/{player_id}", response_model=SimilarResponse)
def similar(
    player_id: str,
    year: Optional[int] = Query(None, description="Restrict to this season"),
    k: int = Query(5, ge=1, le=20, description="Number of neighbors")
):
    subset = emb_df[
        (emb_df.playerID == player_id)
        & ((emb_df.yearID == year) if year is not None else True)
    ]
    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding data for {player_id}"
                   + (f" in {year}" if year is not None else "")
        )

    qidx = int(subset.index.max())
    qvec = features[qidx : qidx + 1]

    D, I = index.search(qvec, k + 1)
    distances, indices = D[0], I[0]

    results: List[SimilarResult] = []
    for dist, idx in zip(distances, indices):
        if idx == qidx:
            continue
        pid, yid = meta[idx]["playerID"], int(meta[idx]["yearID"])
        first_name, last_name = first_names[pid], last_names[pid]
        results.append(SimilarResult(
            playerID=pid,
            nameFirst = first_name,
            nameLast = last_name,
            yearID=yid,
            distance=float(dist)
        ))
        if len(results) >= k:
            break

    return SimilarResponse(
        query_player=player_id,
        query_year=int(emb_df.at[qidx, "yearID"]),
        results=results
    )

@app.get("/players", response_model=List[PlayerSuggestion])
def players(q: str = Query(..., min_length=2, description="Substring to search")):
    mask = players_df["fullName"].str.contains(q, case=False, na=False)
    matches = players_df[mask].head(10)

    return [
        PlayerSuggestion(
            playerID=row.playerID,
            nameFirst=row.nameFirst,
            nameLast=row.nameLast
        )
        for _, row in matches.iterrows()
    ]