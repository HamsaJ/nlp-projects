import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DependencyParsingComponent } from './dependency-parsing.component';

describe('DependencyParsingComponent', () => {
  let component: DependencyParsingComponent;
  let fixture: ComponentFixture<DependencyParsingComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DependencyParsingComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DependencyParsingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
