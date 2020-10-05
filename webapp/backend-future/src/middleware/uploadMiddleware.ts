import { Router } from "express";
import { fileHandler } from "../utils/fileUpload";
const uploadHandler = fileHandler.array("photos", 8);

export const fileCheck = async (requestHandler: Router, req, res, next) => {
  await uploadHandler(req, res, async function (err) {
    if (err) {
      if (err.code === "LIMIT_FILE_SIZE") {
        // await loggingService.create(process.env.FILE_SIZE_ERROR, 5);
        res.status(500).json({ message: process.env.FILE_SIZE_ERROR });
        return;
      }
      res.status(500).json({ message: "" + err });
      return;
    }
    await requestHandler(req, res, next).catch(async (err) => {
      // await loggingService.create(err, 5);
      res.status(500).json({ message: "" + err });
    });
  });
};
