import "reflect-metadata";
import { Server } from "./server/server";

const server = new Server();

server.start(+process.env.PORT || 8000);

// if(certificateOptions !== null){
//     server.startHttpsServer(+process.env.PORT || 8000, process.env.HOST);
// }else {
    
// }