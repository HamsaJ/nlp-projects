import { BrowserModule } from "@angular/platform-browser";
import { NgModule } from "@angular/core";
import { ReactiveFormsModule } from "@angular/forms";
import { RouterModule, Routes } from "@angular/router";
import { AppComponent } from "./app.component";
import { HttpClientModule } from "@angular/common/http";
import { WordSimilarityComponent } from "./word-similarity/word-similarity.component";
import { NamedEntityRecognitionComponent } from "./named-entity-recognition/named-entity-recognition.component";
import { PosTaggingComponent } from "./pos-tagging/pos-tagging.component";
import { DependencyParsingComponent } from "./dependency-parsing/dependency-parsing.component";
import { HomeComponent } from "./home/home.component";
import { SentimentAnalysisComponent } from "./sentiment-analysis/sentiment-analysis.component";
import { CvRankingComponent } from "./cv-ranking/cv-ranking.component";

import { ChartsModule } from "ng2-charts";
import { LayoutModule } from "@angular/cdk/layout";
import { MatToolbarModule } from "@angular/material/toolbar";
import { MatButtonModule } from "@angular/material/button";
import { MatSidenavModule } from "@angular/material/sidenav";
import { MatIconModule } from "@angular/material/icon";
import { MatListModule } from "@angular/material/list";
import { MatGridListModule } from "@angular/material/grid-list";
import { MatCardModule } from "@angular/material/card";
import { MatProgressBarModule } from "@angular/material/progress-bar";
import { MatMenuModule } from "@angular/material/menu";
import { UploadService } from "./upload.service";
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

const appRoutes: Routes = [
  { path: "", component: HomeComponent },
  { path: "lab/wordsim", component: WordSimilarityComponent },
  { path: "lab/ner", component: NamedEntityRecognitionComponent },
  { path: "lab/pos", component: PosTaggingComponent },
  { path: "lab/dependency", component: DependencyParsingComponent },
  { path: "lab/sentan", component: SentimentAnalysisComponent },
  { path: "lab/cvranking", component: CvRankingComponent },
];
@NgModule({
  declarations: [
    AppComponent,
    WordSimilarityComponent,
    NamedEntityRecognitionComponent,
    PosTaggingComponent,
    DependencyParsingComponent,
    HomeComponent,
    SentimentAnalysisComponent,
    CvRankingComponent,
  ],
  imports: [
    MatMenuModule,
    MatProgressBarModule,
    MatCardModule,
    MatGridListModule,
    MatListModule,
    MatIconModule,
    MatSidenavModule,
    MatButtonModule,
    MatToolbarModule,
    LayoutModule,
    ChartsModule,

    BrowserModule,
    HttpClientModule,
    ReactiveFormsModule,
    RouterModule.forRoot(
      appRoutes,
      { enableTracing: false } // <-- debugging purposes only
    ),
    BrowserAnimationsModule,
  ],
  exports: [RouterModule],
  providers: [UploadService],
  bootstrap: [AppComponent],
})
export class AppModule {}
