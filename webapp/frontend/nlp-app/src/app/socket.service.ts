import * as io from 'socket.io-client';

export class SocketService {
    private url = 'http://localhost:8000';
    private socket: SocketIOClient.Socket;    

    constructor() {
        this.socket = io(this.url);
        this.socket.emit('new-message', "Hello world");
    }
}