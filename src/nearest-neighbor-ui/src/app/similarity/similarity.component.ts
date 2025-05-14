import { Component, OnInit }    from '@angular/core';
import { CommonModule }         from '@angular/common';
import { ReactiveFormsModule, FormsModule, FormControl } from '@angular/forms';
import { MatCardModule }        from '@angular/material/card';
import { MatToolbarModule }     from '@angular/material/toolbar';
import { MatFormFieldModule }   from '@angular/material/form-field';
import { MatInputModule }       from '@angular/material/input';
import { MatButtonModule }      from '@angular/material/button';
import { MatTableModule }       from '@angular/material/table';
import { MatAutocompleteModule }from '@angular/material/autocomplete';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { tap, catchError } from 'rxjs/operators';

import {
  SimilarityService,
  SimilarResult,
  PlayerSuggestion
} from '../similarity.service';
import {
  debounceTime,
  distinctUntilChanged,
  filter,
  switchMap,
  finalize,
  startWith
} from 'rxjs/operators';
import { Observable, of } from 'rxjs';

@Component({
  selector: 'app-similarity',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    FormsModule,
    MatCardModule,
    MatToolbarModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatTableModule,
    MatAutocompleteModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './similarity.component.html',
  styleUrls: ['./similarity.component.scss']
})
export class SimilarityComponent implements OnInit {
  // Form controls
  playerCtrl = new FormControl<string>('');
  year?: number;
  k = 5;

  // Autocomplete
  filteredPlayers$: Observable<PlayerSuggestion[]> = of([]);
  loadingPlayers = false;
  selectedPlayerID = '';

  loading = false;
  error = '';
  results: SimilarResult[] = [];
  displayedColumns = ['name','playerID','yearID','distance'];

  constructor(private svc: SimilarityService) {}

  ngOnInit() {
    this.filteredPlayers$ = this.playerCtrl.valueChanges.pipe(
      startWith(''),
      filter(val => typeof val === 'string' && val.length > 1),
      tap(val => console.log('AUTOCOMPLETE PIPELINE:', val)),
      debounceTime(300),
      distinctUntilChanged(),
      switchMap(val => {
        this.loadingPlayers = true;
        return this.svc.getPlayerSuggestions(val as string).pipe(
          finalize(() => this.loadingPlayers = false)
        );
      })
    );
  }

  displayPlayer(player?: PlayerSuggestion): string {
    console.log("gets here?")
    return player
      ? `${player.nameFirst} ${player.nameLast} (${player.playerID})`
      : '';
  }

  onPlayerSelected(player: PlayerSuggestion) {
    this.selectedPlayerID = player.playerID;
  }

  search() {
    if (!this.selectedPlayerID) return;
    this.error = '';
    this.loading = true;
    this.results = [];

    this.svc.getSimilar(this.selectedPlayerID, this.year, this.k)
      .subscribe({
        next: res => {
          this.results = res.results;
          this.loading = false;
        },
        error: err => {
          this.error = err.error?.detail || 'Error fetching data';
          this.loading = false;
        }
      });
  }
}