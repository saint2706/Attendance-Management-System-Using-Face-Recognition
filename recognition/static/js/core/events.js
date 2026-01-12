/**
 * Simple EventBus for cross-module communication.
 *
 * Provides a decoupled way for modules to communicate without direct dependencies.
 * Modules can emit events and subscribe to events from other modules.
 *
 * @module core/events
 *
 * @example
 * import { globalBus } from './core/events.js';
 *
 * // Subscribe to an event
 * globalBus.on('theme:changed', (data) => {
 *   console.log('Theme changed to:', data.theme);
 * });
 *
 * // Emit an event
 * globalBus.emit('theme:changed', { theme: 'dark' });
 *
 * // Unsubscribe
 * const handler = (data) => console.log(data);
 * globalBus.on('someEvent', handler);
 * globalBus.off('someEvent', handler);
 */

/**
 * EventBus class for managing event subscriptions and emissions.
 */
export class EventBus {
    constructor() {
        /**
         * Map of event names to arrays of callback functions
         * @private
         */
        this._events = new Map();

        /**
         * Enable debug logging
         * @type {boolean}
         */
        this.debug = false;
    }

    /**
     * Subscribe to an event.
     *
     * @param {string} event - Event name to subscribe to
     * @param {Function} callback - Function to call when event is emitted
     * @returns {EventBus} Returns this for chaining
     *
     * @example
     * bus.on('user:login', (user) => console.log('User logged in:', user));
     */
    on(event, callback) {
        if (typeof callback !== 'function') {
            throw new TypeError('Callback must be a function');
        }

        if (!this._events.has(event)) {
            this._events.set(event, []);
        }

        this._events.get(event).push(callback);

        if (this.debug) {
            console.log(`[EventBus] Subscribed to '${event}'`);
        }

        return this;
    }

    /**
     * Subscribe to an event, but only fire once.
     * The listener is automatically removed after the first call.
     *
     * @param {string} event - Event name to subscribe to
     * @param {Function} callback - Function to call when event is emitted
     * @returns {EventBus} Returns this for chaining
     */
    once(event, callback) {
        const onceWrapper = (...args) => {
            this.off(event, onceWrapper);
            callback.apply(this, args);
        };

        return this.on(event, onceWrapper);
    }

    /**
     * Unsubscribe from an event.
     *
     * @param {string} event - Event name to unsubscribe from
     * @param {Function} callback - The callback function to remove
     * @returns {EventBus} Returns this for chaining
     *
     * @example
     * const handler = (data) => console.log(data);
     * bus.on('myEvent', handler);
     * bus.off('myEvent', handler); // Unsubscribe
     */
    off(event, callback) {
        if (!this._events.has(event)) {
            return this;
        }

        const callbacks = this._events.get(event);
        const index = callbacks.indexOf(callback);

        if (index !== -1) {
            callbacks.splice(index, 1);

            if (this.debug) {
                console.log(`[EventBus] Unsubscribed from '${event}'`);
            }
        }

        // Clean up empty event arrays
        if (callbacks.length === 0) {
            this._events.delete(event);
        }

        return this;
    }

    /**
     * Remove all listeners for a specific event, or all events if no event specified.
     *
     * @param {string} [event] - Optional event name. If not provided, clears all events.
     * @returns {EventBus} Returns this for chaining
     */
    clear(event) {
        if (event) {
            this._events.delete(event);

            if (this.debug) {
                console.log(`[EventBus] Cleared all listeners for '${event}'`);
            }
        } else {
            this._events.clear();

            if (this.debug) {
                console.log('[EventBus] Cleared all listeners');
            }
        }

        return this;
    }

    /**
     * Emit an event, calling all subscribed callbacks.
     *
     * @param {string} event - Event name to emit
     * @param {*} data - Data to pass to callbacks
     * @returns {EventBus} Returns this for chaining
     *
     * @example
     * bus.emit('data:loaded', { items: [...] });
     */
    emit(event, data) {
        if (!this._events.has(event)) {
            if (this.debug) {
                console.log(`[EventBus] No listeners for '${event}'`);
            }
            return this;
        }

        const callbacks = this._events.get(event);

        if (this.debug) {
            console.log(`[EventBus] Emitting '${event}' to ${callbacks.length} listeners`, data);
        }

        // Create a copy of callbacks array to avoid issues if callbacks modify the array
        [...callbacks].forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`[EventBus] Error in event handler for '${event}':`, error);
            }
        });

        return this;
    }

    /**
     * Get the number of listeners for an event.
     *
     * @param {string} event - Event name
     * @returns {number} Number of listeners
     */
    listenerCount(event) {
        return this._events.has(event) ? this._events.get(event).length : 0;
    }

    /**
     * Get all event names that have listeners.
     *
     * @returns {string[]} Array of event names
     */
    eventNames() {
        return Array.from(this._events.keys());
    }
}

/**
 * Global event bus instance for use across the application.
 * Import this to communicate between modules.
 *
 * @type {EventBus}
 *
 * @example
 * import { globalBus } from './core/events.js';
 * globalBus.on('myEvent', (data) => console.log(data));
 */
export const globalBus = new EventBus();

// Enable debug mode in development
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Uncomment to enable debug logging
    // globalBus.debug = true;
}
