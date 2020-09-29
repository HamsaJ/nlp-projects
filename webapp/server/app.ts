import * as express from "express";
import * as multer from "multer";
import * as cors from "cors";
import * as fs from "fs";
import * as path from "path";
import * as Loki from "lokijs";
import { Collection } from "lokijs";
import * as del from "del";

const imageFilter = function (req, file, cb) {
  // accept image only
  if (!file.originalname.match(/\.(jpg|jpeg|png|gif)$/)) {
    return cb(new Error("Only image files are allowed!"), false);
  }
  cb(null, true);
};

const loadCollection = function (colName, db: Loki): Promise<Collection<any>> {
  return new Promise((resolve) => {
    db.loadDatabase({}, () => {
      const _collection =
        db.getCollection(colName) || db.addCollection(colName);
      resolve(_collection);
    });
  });
};

const cleanFolder = function (folderPath) {
  // delete files inside folder but not the folder itself
  del.sync([`${folderPath}/**`, `!${folderPath}`]);
};

// setup
const DB_NAME = "db.json";
const COLLECTION_NAME = "images";
const UPLOAD_PATH = "uploads";
const upload = multer({ dest: `${UPLOAD_PATH}/`, fileFilter: imageFilter });
const db = new Loki(`${UPLOAD_PATH}/${DB_NAME}`, { persistenceMethod: "fs" });

// optional: clean all data before start
// cleanFolder(UPLOAD_PATH);

// app
const app = express();
app.use(cors());

app.get("/", async (req, res) => {
  // default route
  res.send(`
        <h1>Demo file upload</h1>
        <p>Please refer to <a href="https://scotch.io/tutorials/express-file-uploads-with-multer">my tutorial</a> for details.</p>
        <ul>
            <li>GET /images   - list all upload images</li>
            <li>GET /images/{id} - get one uploaded image</li>
            <li>POST /profile - handle single image upload</li>
            <li>POST /photos/upload - handle multiple images upload</li>
        </ul>
    `);
});

app.post("/profile", upload.single("avatar"), async (req, res) => {
  try {
    const col = await loadCollection(COLLECTION_NAME, db);
    const data = col.insert(req.file);

    db.saveDatabase();
    res.send({
      id: data.$loki,
      fileName: data.filename,
      originalName: data.originalname,
    });
  } catch (err) {
    res.sendStatus(400);
  }
});

app.post("/photos/upload", upload.array("photos", 12), async (req, res) => {
  try {
    console.log("REQUEST +++++++++++++++++", req.files);
    const col = await loadCollection(COLLECTION_NAME, db);
    let data = [].concat(col.insert(req.files));

    db.saveDatabase();
    res.send(
      data.map((x) => ({
        id: x.$loki,
        fileName: x.filename,
        originalName: x.originalname,
      }))
    );
  } catch (err) {
    res.sendStatus(400);
  }
});

app.get("/images", async (req, res) => {
  try {
    const col = await loadCollection(COLLECTION_NAME, db);
    res.send(col.data);
  } catch (err) {
    res.sendStatus(400);
  }
});

app.get("/images/:id", async (req: any, res) => {
  try {
    const col = await loadCollection(COLLECTION_NAME, db);
    const result = col.get(req.params.id);

    if (!result) {
      res.sendStatus(404);
      return;
    }

    res.setHeader("Content-Type", result.mimetype);
    fs.createReadStream(path.join(UPLOAD_PATH, result.filename)).pipe(res);
  } catch (err) {
    res.sendStatus(400);
  }
});

app.listen(3000, function () {
  console.log("listening on port 3000!");
});
