import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { NamedEntityRecognitionComponent } from './named-entity-recognition.component';

describe('NamedEntityRecognitionComponent', () => {
  let component: NamedEntityRecognitionComponent;
  let fixture: ComponentFixture<NamedEntityRecognitionComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ NamedEntityRecognitionComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(NamedEntityRecognitionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
