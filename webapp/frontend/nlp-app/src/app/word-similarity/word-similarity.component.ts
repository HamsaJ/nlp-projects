import { Component, OnInit } from "@angular/core";
import {
  FormControl,
  FormGroup,
  FormBuilder,
  Validators,
} from "@angular/forms";
import { HttpClient, HttpHeaders } from "@angular/common/http";

@Component({
  selector: "app-word-similarity",
  templateUrl: "./word-similarity.component.html",
  styleUrls: ["./word-similarity.component.css"],
})
export class WordSimilarityComponent implements OnInit {
  ngOnInit() {}
  endpoint = "http://localhost:8000/api/wordsim";
  headers = { "Content-Type": "application/json" };
  data: any;
  result: any;
  found: boolean;

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
          this.result = Array.from(
            result["response"]["body"]["wordSimilarity"]
          );
          this.found = true;
          console.log(this.result);
        }
      });
  }
}
