import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { CvRankingComponent } from './cv-ranking.component';

describe('CvRankingComponent', () => {
  let component: CvRankingComponent;
  let fixture: ComponentFixture<CvRankingComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ CvRankingComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(CvRankingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
