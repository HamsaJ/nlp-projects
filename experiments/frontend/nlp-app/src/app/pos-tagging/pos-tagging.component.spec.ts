import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { PosTaggingComponent } from './pos-tagging.component';

describe('PosTaggingComponent', () => {
  let component: PosTaggingComponent;
  let fixture: ComponentFixture<PosTaggingComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ PosTaggingComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(PosTaggingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
