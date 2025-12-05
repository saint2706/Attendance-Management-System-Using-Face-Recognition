/**
 * Centralized configuration for all frontend modules.
 * 
 * This module provides a single source of truth for configuration values
 * that were previously hardcoded across multiple files.
 * 
 * @module core/config
 */

/**
 * Theme management configuration
 */
export const THEME_CONFIG = {
    STORAGE_KEY: 'attendance-theme',
    DARK_CLASS: 'theme-dark',
    LIGHT_CLASS: 'theme-light',
    DEFAULT_THEME: 'light'
};

/**
 * Table enhancement configuration
 */
export const TABLE_CONFIG = {
    CSV_FILENAME_PREFIX: 'attendance_export',
    CSV_DELIMITER: ',',
    SEARCH_DEBOUNCE_MS: 300,
    SORT_ICON_ASC: 'bi-arrow-up',
    SORT_ICON_DESC: 'bi-arrow-down',
    SORT_ICON_DEFAULT: 'bi-arrow-down-up'
};

/**
 * Offline attendance queue configuration
 */
export const OFFLINE_CONFIG = {
    DB_NAME: 'attendance-offline',
    STORE_NAME: 'attendance-queue',
    ATTENDANCE_ENDPOINTS: [
        '/mark_your_attendance',
        '/mark_your_attendance_out'
    ],
    RETRY_DELAY_MS: 2000,
    MAX_RETRIES: 3
};

/**
 * Polling and real-time update configuration
 */
export const POLLING_CONFIG = {
    ATTENDANCE_LOG_INTERVAL_MS: 5000,
    RETRY_INTERVAL_MS: 10000,
    MAX_POLLING_ERRORS: 5
};

/**
 * Alert auto-dismiss configuration
 */
export const ALERT_CONFIG = {
    AUTO_DISMISS_MS: 5000,
    FADE_OUT_MS: 300
};

/**
 * Form validation configuration
 */
export const FORM_CONFIG = {
    VALIDATION_DEBOUNCE_MS: 500
};

/**
 * API endpoint configuration
 * These can be overridden at runtime if needed
 */
export const API_CONFIG = {
    ATTENDANCE_FEED_URL: '/api/attendance/feed',
    MARK_ATTENDANCE_IN_URL: '/mark_your_attendance',
    MARK_ATTENDANCE_OUT_URL: '/mark_your_attendance_out'
};

/**
 * Get a configuration value by path
 * Example: getConfig('THEME_CONFIG.STORAGE_KEY')
 * 
 * @param {string} path - Dot-separated path to config value
 * @returns {*} The configuration value or undefined if not found
 */
export function getConfig(path) {
    const parts = path.split('.');
    let current = { THEME_CONFIG, TABLE_CONFIG, OFFLINE_CONFIG, POLLING_CONFIG, ALERT_CONFIG, FORM_CONFIG, API_CONFIG };

    for (const part of parts) {
        if (current === undefined || current === null) {
            return undefined;
        }
        current = current[part];
    }

    return current;
}

/**
 * Merge user configuration with defaults
 * 
 * @param {Object} userConfig - User-provided configuration
 * @param {Object} defaultConfig - Default configuration
 * @returns {Object} Merged configuration
 */
export function mergeConfig(userConfig, defaultConfig) {
    return { ...defaultConfig, ...userConfig };
}
