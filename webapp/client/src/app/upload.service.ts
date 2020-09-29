import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root',
})
export class UploadService {
  SERVER_URL: string = 'http://127.0.0.1:5000/';
  constructor(private httpClient: HttpClient) {}

  public upload(formData) {
    console.log(formData);

    return this.httpClient.post<any>(this.SERVER_URL, formData, {
      reportProgress: true,
      observe: 'events',
    });
  }
}
