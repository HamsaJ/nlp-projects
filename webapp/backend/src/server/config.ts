import { Router } from "express";

export interface HttpServer {
    
    /**
     * 
     * @param url 
     * @param requestHandler 
     * @param file 
     * @param auth 
     */
    post(url: string, requestHandler: Router): void;

    /**
     * 
     * @param url 
     * @param requestHandler 
     * @param file 
     * @param auth 
     */
    get(url: string, requestHandler: Router): void;

    /**
     * 
     * @param url 
     * @param requestHandler 
     * @param file 
     * @param auth 
     */
    put(url: string, requestHandler: Router, file?:any, auth?:any): void;

    /**
     * 
     * @param url 
     * @param requestHandler 
     * @param file 
     * @param auth 
     */
    del(url: string, requestHandler: Router, file?:any, auth?:any): void;
}