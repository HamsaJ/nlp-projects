import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

// app-routing.module.ts
import { DashComponent } from './dash/dash.component';

const routes: Routes = [{ path: 'dashboard', component: DashComponent }];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
