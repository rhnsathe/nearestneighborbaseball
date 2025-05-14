import { TestBed } from '@angular/core/testing';

import { SimilarityService } from './similarity.service';

describe('SimilarityService', () => {
  let service: SimilarityService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SimilarityService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
