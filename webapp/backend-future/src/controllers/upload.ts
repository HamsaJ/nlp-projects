import { Controller } from "./controller";
import { HttpServer } from "../server/config";
import { Request, Response } from "express";
import { nlpengineService } from "../services/nlpengine";
import { fileHandler } from "../utils/fileUpload";

export class UploadController implements Controller {
  private uploadHandler = fileHandler.array("photos", 8);
  /**
   *
   * @param httpServer Server interface
   */
  public initialize(httpServer: HttpServer): void {
    httpServer.post("/upload", this.upload.bind(this), this.uploadHandler);
    // httpServer.put('/api/account/:id', this.update.bind(this));
  }
  /**
   *
   * @param req http request to get a user by id
   * @param res http response to return a user by id
   */
  private async upload(req: Request, res: Response): Promise<Response> {
    try {
      // console.log(req.files);
      return res.status(201).json({
        message: "File has successfully been uploaded",
      });
    } catch (error) {
      console.log("ERROR: ", error);
      return res.status(500).json({ message: "Something went wrong" });
    }
  }
}
