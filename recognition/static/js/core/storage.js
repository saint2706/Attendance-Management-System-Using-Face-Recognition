/**
 * Storage abstraction layer for localStorage and IndexedDB.
 * 
 * Provides consistent interfaces for browser storage with error handling,
 * making storage operations easier to test and mock.
 * 
 * @module core/storage
 */

/**
 * LocalStorage adapter with consistent error handling.
 * 
 * @example
 * import { localStore } from './core/storage.js';
 * 
 * localStore.set('user', { name: 'John' });
 * const user = localStore.get('user');
 * localStore.remove('user');
 */
export class LocalStorageAdapter {
    /**
     * Get a value from localStorage.
     * 
     * @param {string} key - Storage key
     * @param {*} [defaultValue=null] - Default value if key doesn't exist
     * @returns {*} Parsed value or default
     */
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);

            if (item === null) {
                return defaultValue;
            }

            // Try to parse as JSON, fall back to raw string
            try {
                return JSON.parse(item);
            } catch {
                return item;
            }
        } catch (error) {
            console.error(`[LocalStorage] Error getting '${key}':`, error);
            return defaultValue;
        }
    }

    /**
     * Set a value in localStorage.
     * 
     * @param {string} key - Storage key
     * @param {*} value - Value to store (will be JSON stringified)
     * @returns {boolean} True if successful
     */
    set(key, value) {
        try {
            const serialized = typeof value === 'string' ? value : JSON.stringify(value);
            localStorage.setItem(key, serialized);
            return true;
        } catch (error) {
            console.error(`[LocalStorage] Error setting '${key}':`, error);
            return false;
        }
    }

    /**
     * Remove a value from localStorage.
     * 
     * @param {string} key - Storage key
     * @returns {boolean} True if successful
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error(`[LocalStorage] Error removing '${key}':`, error);
            return false;
        }
    }

    /**
     * Clear all localStorage items.
     * 
     * @returns {boolean} True if successful
     */
    clear() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('[LocalStorage] Error clearing:', error);
            return false;
        }
    }

    /**
     * Check if a key exists in localStorage.
     * 
     * @param {string} key - Storage key
     * @returns {boolean} True if key exists
     */
    has(key) {
        return localStorage.getItem(key) !== null;
    }
}

/**
 * IndexedDB adapter with Promise-based API.
 * 
 * @example
 * import { IndexedDBAdapter } from './core/storage.js';
 * 
 * const db = new IndexedDBAdapter('myDB', 'myStore');
 * await db.set('user', { name: 'John' });
 * const user = await db.get('user');
 */
export class IndexedDBAdapter {
    /**
     * @param {string} dbName - Database name
     * @param {string} storeName - Object store name
     * @param {number} [version=1] - Database version
     */
    constructor(dbName, storeName, version = 1) {
        this.dbName = dbName;
        this.storeName = storeName;
        this.version = version;
        this._dbPromise = null;
    }

    /**
     * Open database connection.
     * 
     * @private
     * @returns {Promise<IDBDatabase>} Database instance
     */
    _openDB() {
        if (this._dbPromise) {
            return this._dbPromise;
        }

        this._dbPromise = new Promise((resolve, reject) => {
            if (!('indexedDB' in window)) {
                reject(new Error('IndexedDB not supported'));
                return;
            }

            const request = indexedDB.open(this.dbName, this.version);

            request.onupgradeneeded = () => {
                const db = request.result;
                if (!db.objectStoreNames.contains(this.storeName)) {
                    db.createObjectStore(this.storeName, { keyPath: 'id', autoIncrement: true });
                }
            };

            request.onsuccess = () => resolve(request.result);
            request.onerror = () => {
                this._dbPromise = null;
                reject(request.error);
            };
        });

        return this._dbPromise;
    }

    /**
     * Get a value from IndexedDB.
     * 
     * @param {string|number} key - Storage key
     * @returns {Promise<*>} The stored value or undefined
     */
    async get(key) {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readonly');
                const store = transaction.objectStore(this.storeName);
                const request = store.get(key);

                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        } catch (error) {
            console.error(`[IndexedDB] Error getting key '${key}':`, error);
            return undefined;
        }
    }

    /**
     * Get all values from the store.
     * 
     * @returns {Promise<Array>} Array of all stored values
     */
    async getAll() {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readonly');
                const store = transaction.objectStore(this.storeName);
                const request = store.getAll();

                request.onsuccess = () => resolve(request.result || []);
                request.onerror = () => reject(request.error);
            });
        } catch (error) {
            console.error('[IndexedDB] Error getting all:', error);
            return [];
        }
    }

    /**
     * Set a value in IndexedDB.
     * 
     * @param {*} value - Value to store (must have 'id' property or use auto-increment)
     * @returns {Promise<boolean>} True if successful
     */
    async set(value) {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readwrite');
                const store = transaction.objectStore(this.storeName);
                const request = store.add(value);

                transaction.oncomplete = () => resolve(true);
                transaction.onerror = () => reject(transaction.error);
            });
        } catch (error) {
            console.error('[IndexedDB] Error setting value:', error);
            return false;
        }
    }

    /**
     * Update a value in IndexedDB.
     * 
     * @param {*} value - Value to update (must have existing 'id')
     * @returns {Promise<boolean>} True if successful
     */
    async update(value) {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readwrite');
                const store = transaction.objectStore(this.storeName);
                const request = store.put(value);

                transaction.oncomplete = () => resolve(true);
                transaction.onerror = () => reject(transaction.error);
            });
        } catch (error) {
            console.error('[IndexedDB] Error updating value:', error);
            return false;
        }
    }

    /**
     * Remove a value from IndexedDB by key.
     * 
     * @param {string|number} key - Storage key
     * @returns {Promise<boolean>} True if successful
     */
    async remove(key) {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readwrite');
                const store = transaction.objectStore(this.storeName);
                const request = store.delete(key);

                transaction.oncomplete = () => resolve(true);
                transaction.onerror = () => reject(transaction.error);
            });
        } catch (error) {
            console.error(`[IndexedDB] Error removing key '${key}':`, error);
            return false;
        }
    }

    /**
     * Clear all values from the store.
     * 
     * @returns {Promise<boolean>} True if successful
     */
    async clear() {
        try {
            const db = await this._openDB();

            return new Promise((resolve, reject) => {
                const transaction = db.transaction(this.storeName, 'readwrite');
                const store = transaction.objectStore(this.storeName);
                const request = store.clear();

                transaction.oncomplete = () => resolve(true);
                transaction.onerror = () => reject(transaction.error);
            });
        } catch (error) {
            console.error('[IndexedDB] Error clearing store:', error);
            return false;
        }
    }

    /**
     * Close the database connection.
     */
    close() {
        if (this._dbPromise) {
            this._dbPromise.then(db => db.close());
            this._dbPromise = null;
        }
    }
}

/**
 * Global localStorage instance for convenience.
 * 
 * @type {LocalStorageAdapter}
 * 
 * @example
 * import { localStore } from './core/storage.js';
 * localStore.set('theme', 'dark');
 */
export const localStore = new LocalStorageAdapter();
