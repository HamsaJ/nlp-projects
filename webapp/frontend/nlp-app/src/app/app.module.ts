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
import { SocketService } from "./socket.service";
import { SentimentAnalysisComponent } from "./sentiment-analysis/sentiment-analysis.component";

const appRoutes: Routes = [
  { path: "", component: HomeComponent },
  { path: "lab/wordsim", component: WordSimilarityComponent },
  { path: "lab/ner", component: NamedEntityRecognitionComponent },
  { path: "lab/pos", component: PosTaggingComponent },
  { path: "lab/dependency", component: DependencyParsingComponent },
  { path: "lab/sentan", component: SentimentAnalysisComponent },
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
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    ReactiveFormsModule,
    RouterModule.forRoot(
      appRoutes,
      { enableTracing: false } // <-- debugging purposes only
    ),
  ],
  exports: [RouterModule],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
