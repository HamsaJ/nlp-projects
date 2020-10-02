import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, FormBuilder, Validators} from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import * as io from 'socket.io-client';
import { SocketService } from './socket.service'
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  ngOnInit(): void {
    
  }
  title = 'NLP2GO';

  constructor() {
    try {
      var url = 'http://localhost:8000';
      var socket: SocketIOClient.Socket; 
      socket = io(url);
      socket.emit('new-message', "Hello world");
    } catch (error) {
      console.log(error)
    }

    console.log('created main app')
   }
}
