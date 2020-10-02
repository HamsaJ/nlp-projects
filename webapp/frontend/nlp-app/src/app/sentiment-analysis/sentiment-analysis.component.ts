import { Component, OnInit } from "@angular/core";
import {
  FormControl,
  FormGroup,
  FormBuilder,
  Validators,
} from "@angular/forms";
import { HttpClient, HttpHeaders } from "@angular/common/http";

@Component({
  selector: "app-sentiment-analysis",
  templateUrl: "./sentiment-analysis.component.html",
  styleUrls: ["./sentiment-analysis.component.css"],
})
export class SentimentAnalysisComponent implements OnInit {
  ngOnInit() {}
  endpoint = "http://localhost:8000/api/sentan";
  headers = { "Content-Type": "application/json" };
  data: any;
  result: any;
  found: boolean;
  isPositive: boolean;

  angularForm = new FormGroup({
    data: new FormControl(),
  });

  constructor(private fb: FormBuilder, private httpClient: HttpClient) {
    this.createForm();
  }

  createForm() {
    this.angularForm = this.fb.group({
      name: ["", Validators.required],
    });
  }

  onNameKeyUp(event: any) {
    this.data = event.target.value;
    this.found = false;
    this.isPositive = false;
  }

  sendText() {
    console.log("sending request", {
      data: this.data.trim(),
    });
    this.httpClient
      .post(this.endpoint, {
        data: this.data,
      })
      .subscribe((result: any[]) => {
        if (result.length !== 0) {
          this.result = Array.from(result["response"]["body"]["sentAnalysis"]);

          if (this.result[0]["label"] === "POSITIVE") {
            console.log("LABEL IS POSITIVE");
            this.isPositive = true;
          } else if (this.result[0]["label"] === "NEGATIVE") {
            this.isPositive = false;
          }

          this.found = true;
          console.log(this.result);
        }
      });
  }
}
