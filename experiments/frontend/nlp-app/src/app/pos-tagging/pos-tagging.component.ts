import { Component, OnInit } from "@angular/core";
import {
  FormControl,
  FormGroup,
  FormBuilder,
  Validators,
} from "@angular/forms";
import { HttpClient, HttpHeaders } from "@angular/common/http";

@Component({
  selector: "app-pos-tagging",
  templateUrl: "./pos-tagging.component.html",
  styleUrls: ["./pos-tagging.component.css"],
})
export class PosTaggingComponent implements OnInit {
  ngOnInit() {}
  endpoint = "http://localhost:8000/nlp/pos";
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
      text: this.data.trim(),
    });
    this.httpClient
      .post(this.endpoint, {
        text: this.data,
      })
      .subscribe((result: any[]) => {
        if (result.length !== 0) {
          this.result = Array.from(result["response"]["body"]["posTags"]);
          this.found = true;
          console.log(this.result);
        }
      });
  }
}
