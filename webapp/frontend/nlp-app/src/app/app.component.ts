import { Component, OnInit } from "@angular/core";
import {
  FormControl,
  FormGroup,
  FormBuilder,
  Validators,
} from "@angular/forms";
import { HttpClient, HttpHeaders } from "@angular/common/http";
@Component({
  selector: "app-root",
  templateUrl: "./app.component.html",
  styleUrls: ["./app.component.css"],
})
export class AppComponent implements OnInit {
  ngOnInit(): void {}
  title = "NLP-LAB";

  constructor() {
    try {
      var url = "http://localhost:8000";
    } catch (error) {
      console.log(error);
    }

    console.log("created main app");
  }
}
