import { HttpServer } from "../server/config";

/**
 * Controller Interface 
 * 
 * with an initialisation method which 
 * all Controller classes must implement
 */

export interface Controller {
    /**
     * 
     * @param httpServer 
     */
    initialize(httpServer: HttpServer): void;
}