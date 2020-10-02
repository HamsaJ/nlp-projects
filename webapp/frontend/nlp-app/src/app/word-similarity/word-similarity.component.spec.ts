import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { WordSimilarityComponent } from './word-similarity.component';

describe('WordSimilarityComponent', () => {
  let component: WordSimilarityComponent;
  let fixture: ComponentFixture<WordSimilarityComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ WordSimilarityComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(WordSimilarityComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
