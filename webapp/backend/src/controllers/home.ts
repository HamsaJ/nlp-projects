import { Controller } from "./controller";
import { HttpServer } from "../server/config";
import { Request, Response } from "express";
import * as path from "path";

export class HomeController implements Controller {
    /**
     * 
     * @param httpServer Server interface
     */
    public initialize(httpServer: HttpServer): void {
        httpServer.get('/', this.indexPage.bind(this));
        // httpServer.put('/api/account/:id', this.update.bind(this));
    }
    /**
     * 
     * @param req http request to get a user by id
     * @param res http response to return a user by id
     */
    private async indexPage(req: Request, res: Response): Promise<void> {
        res.send('Welcome to NLP2GO backend');  
    } 
}