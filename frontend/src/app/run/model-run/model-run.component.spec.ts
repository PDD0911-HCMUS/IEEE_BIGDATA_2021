import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelRunComponent } from './model-run.component';

describe('ModelRunComponent', () => {
  let component: ModelRunComponent;
  let fixture: ComponentFixture<ModelRunComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ModelRunComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ModelRunComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
