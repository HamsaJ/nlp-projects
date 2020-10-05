import "reflect-metadata";
import request = require("request");

const NLP_HOST = "http://localhost:5000/";
export class NlpengineService {
  public async wordSim(data: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      console.log(data);
      const clientServerOptions = {
        uri: NLP_HOST + "wordsim",
        json: { data: data["data"] },
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      };

      // Make a request to the predictions api
      request(clientServerOptions, async (err, response) => {
        if (err) {
          reject(err);
        }
        try {
          if (response.statusCode !== 200) {
            throw {
              message: `There was a problem with the prediction service! : ${response.body.error}`,
            };
          }
          resolve(response);
        } catch (err) {
          reject({ error: response.body.error });
        }
      });
    });
  }

  public async wordDep(data: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      console.log(data);
      const clientServerOptions = {
        uri: NLP_HOST + "dependency",
        json: { data: data["data"] },
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      };

      // Make a request to the predictions api
      request(clientServerOptions, async (err, response) => {
        if (err) {
          reject(err);
        }
        try {
          if (response.statusCode !== 200) {
            throw {
              message: `There was a problem with the prediction service! : ${response.body.error}`,
            };
          }
          resolve(response);
        } catch (err) {
          reject({ error: response.body.error });
        }
      });
    });
  }

  public async ner(data: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      console.log(data);
      const clientServerOptions = {
        uri: NLP_HOST + "ner",
        json: { data: data["data"] },
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      };

      // Make a request to the predictions api
      request(clientServerOptions, async (err, response) => {
        if (err) {
          reject(err);
        }
        try {
          if (response.statusCode !== 200) {
            throw {
              message: `There was a problem with the prediction service! : ${response.body.error}`,
            };
          }
          resolve(response);
        } catch (err) {
          reject({ error: response.body.error });
        }
      });
    });
  }

  public async pos(data: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      console.log(data);
      const clientServerOptions = {
        uri: NLP_HOST + "postag",
        json: { data: data["data"] },
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      };

      // Make a request to the predictions api
      request(clientServerOptions, async (err, response) => {
        if (err) {
          reject(err);
        }
        try {
          if (response.statusCode !== 200) {
            throw {
              message: `There was a problem with the prediction service! : ${response.body.error}`,
            };
          }
          resolve(response);
        } catch (err) {
          reject({ error: response.body.error });
        }
      });
    });
  }

  public async sentAn(data: string): Promise<any> {
    return new Promise(async (resolve, reject) => {
      console.log(data);
      const clientServerOptions = {
        uri: NLP_HOST + "sentan",
        json: { data: data["data"] },
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      };

      // Make a request to the predictions api
      request(clientServerOptions, async (err, response) => {
        if (err) {
          reject(err);
        }
        try {
          if (response.statusCode !== 200) {
            throw {
              message: `There was a problem with the prediction service! : ${response.body.error}`,
            };
          }
          resolve(response);
        } catch (err) {
          reject({ error: response.body.error });
        }
      });
    });
  }
}

export const nlpengineService = new NlpengineService();
