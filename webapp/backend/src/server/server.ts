import { HttpServer } from "./config";
import { Router } from "express";
import { CONTROLLERS } from "../controllers";
import  * as express from "express";
import { accessControlCors } from "../middleware/accessControlCors";
import * as cors from "cors";
import bodyParser = require("body-parser");
/**
 * 
 */
export class Server implements HttpServer {
    private server: express.Application;
    private httpsServer;
    
    constructor() {
        this.server = express();
        this.server.use(cors());
        this.server.use(bodyParser.json({limit: '150mb'}));
        this.server.use(bodyParser.urlencoded({ limit: '150mb', extended: true }));
        CONTROLLERS.forEach(controller => controller.initialize(this));
    }
    public post(url: string, requestHandler: Router): void {
        this.addRoute('post', url, requestHandler);
    }

    public get(url: string, requestHandler: Router): void {
        this.addRoute('get', url, requestHandler);
    }
    
    public put(url: string, requestHandler: Router): void {
        this.addRoute('put', url, requestHandler);
    }

    public del(url: string, requestHandler: Router): void {
        this.addRoute('del', url, requestHandler);
    }

    // TODO: Clean up the function below! 
    /**
     * 
     * @param method 
     * @param url 
     * @param requestHandler 
     * @param fileHandler 
     * @param auth 
     */
    private addRoute(method: 'post' | 'get' | 'put' | 'del', url: string, requestHandler: Router){
        this.server[method](`/api${url}`, async (req, res, next) => {
            try {
                await requestHandler(req, res, next);
                
            }
            catch(e) {
                console.log(e);
            }
        });
        console.log(`Added route ${method.toUpperCase()}: /api${url}`);
    }
    /**
     * 
     * @param port 
     */
    public start(port: number, host?:string): number {

        // Add controllers
        CONTROLLERS.forEach(controller => controller.initialize(this));

        this.server.listen(port, async () => {
           console.log(`server started on port: ${port}`);
        });
        return 1;
    }

    // /**
    //  * 
    //  * @param port 
    //  */
    // public async startHttpsServer(port: number, host:string): Promise<number> {
    //     const certificateOptions = certOptions();
    //     CONTROLLERS.forEach(controller => controller.initialize(this));
    //     https.createServer(certificateOptions, this.server)
    //     .listen(process.env.PORT || 8000)
    //     logger.debug(`Backend Server is up and running on Host: ${host} and Port: ${port}`);
    //     // await loggingService.create(`Backend Server is up and running on Host: ${host} and Port: ${port}`, 2);
    //     return 1;
    // }
}