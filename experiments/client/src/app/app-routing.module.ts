import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

// app-routing.module.ts
import { DashComponent } from './dash/dash.component';
import { HomeComponent } from './home/home.component';

const routes: Routes = [
  { path: 'dashboard', component: DashComponent },
  { path: 'home', component: HomeComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
