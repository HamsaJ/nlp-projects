import { Controller } from "./controller";
import { HttpServer } from "../server/config";
import { Request, Response } from "express";
import { nlpengineService } from "../services/nlpengine";

export class DependencyController implements Controller {
    /**
     * 
     * @param httpServer Server interface
     */
    public initialize(httpServer: HttpServer): void {
        httpServer.post('/dependency', this.wordDep.bind(this));
        // httpServer.put('/api/account/:id', this.update.bind(this));
    }
    /**
     * 
     * @param req http request to get a user by id
     * @param res http response to return a user by id
     */
    private async wordDep(req: Request, res: Response): Promise<Response> {
        const data = await req.body
        return await nlpengineService.wordDep(data)
            .then( response => {
                return res.status(200).json({
                    success : true,
                    response
                })
            })
            .catch( err => {
                console.log(JSON.stringify(err))
                return res.status(500).json({
                    success : false,
                    error : JSON.stringify(err)
                })               
            })
    } 
}