import multer = require("multer");

const PATH = process.cwd() + "/upload/";

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    console.log(req);
    cb(null, PATH);
  },
  filename: (req, file, cb) => {
    console.log(req);
    const fileName = file.originalname.toLowerCase().split(" ").join("-");
    cb(null, fileName);
  },
});

export const fileHandler = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    console.log(req);
    if (
      file.mimetype == "image/png" ||
      file.mimetype == "image/jpg" ||
      file.mimetype == "image/jpeg" ||
      file.mimetype == "image/gif"
    ) {
      cb(null, true);
    } else {
      cb(null, false);
      return cb(new Error("Allowed only .png, .jpg, .jpeg and .gif"));
    }
  },
});
