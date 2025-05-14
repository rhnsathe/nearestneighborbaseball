// src/app/similarity.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SimilarResult {
  player_id: string;
  nameFirst: string;
  nameLast: string;
  year: number;
  distance: number;
}

export interface SimilarResponse {
  query_player: string;
  query_year: number;
  results: SimilarResult[];
}

export interface PlayerSuggestion {
  player_id: number;
  nameFirst: string;
  nameLast: string;
}

@Injectable({ providedIn: 'root' })
export class SimilarityService {
  private baseUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  getSimilar(
    player_id: number,
    year?: number,
    k: number = 5
  ): Observable<SimilarResponse> {
    let params = new HttpParams().set('k', k.toString());
    if (year != null) {
      params = params.set('year', year.toString());
    }
    return this.http.get<SimilarResponse>(
      `${this.baseUrl}/similar_v2/${player_id}`,
      { params }
    );
  }

  getPlayerSuggestions(query: string): Observable<PlayerSuggestion[]> {
    return this.http.get<PlayerSuggestion[]>(
      `${this.baseUrl}/players_v2`,
      { params: new HttpParams().set('q', query) }
    );
  }
}
