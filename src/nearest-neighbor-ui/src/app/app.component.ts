import { Component }           from '@angular/core';
import { SimilarityComponent } from './similarity/similarity.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [SimilarityComponent],
  template: `<app-similarity></app-similarity>`
})
export class AppComponent {}
