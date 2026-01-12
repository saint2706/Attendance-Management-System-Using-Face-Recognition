/**
 * Theme management module for dark/light mode toggle.
 *
 * @module ui/theme
 */

import { THEME_CONFIG } from '../core/config.js';
import { localStore } from '../core/storage.js';
import { globalBus } from '../core/events.js';

/**
 * Manages the application's theme (dark/light mode).
 * Persists user preference to localStorage and emits events on theme change.
 */
export class ThemeManager {
    constructor() {
        this.toggleBtn = null;
    }

    /**
     * Initialize the theme manager.
     * Loads saved theme preference and sets up toggle button.
     */
    init() {
        // Load saved theme preference or default to light
        const savedTheme = localStore.get(THEME_CONFIG.STORAGE_KEY, THEME_CONFIG.DEFAULT_THEME);

        if (savedTheme === 'dark') {
            this.enableDarkMode(false); // false = don't emit event on init
        }

        // Set up theme toggle button
        this.toggleBtn = document.getElementById('theme-toggle');
        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
            this.updateToggleIcon();
        }
    }

    /**
     * Check if dark mode is currently enabled.
     *
     * @returns {boolean} True if dark mode is active
     */
    isDarkMode() {
        return document.documentElement.classList.contains(THEME_CONFIG.DARK_CLASS);
    }

    /**
     * Enable dark mode.
     *
     * @param {boolean} [emitEvent=true] - Whether to emit theme:changed event
     */
    enableDarkMode(emitEvent = true) {
        document.documentElement.classList.add(THEME_CONFIG.DARK_CLASS);
        localStore.set(THEME_CONFIG.STORAGE_KEY, 'dark');
        this.updateToggleIcon();

        if (emitEvent) {
            globalBus.emit('theme:changed', { theme: 'dark' });
        }
    }

    /**
     * Enable light mode.
     *
     * @param {boolean} [emitEvent=true] - Whether to emit theme:changed event
     */
    enableLightMode(emitEvent = true) {
        document.documentElement.classList.remove(THEME_CONFIG.DARK_CLASS);
        localStore.set(THEME_CONFIG.STORAGE_KEY, 'light');
        this.updateToggleIcon();

        if (emitEvent) {
            globalBus.emit('theme:changed', { theme: 'light' });
        }
    }

    /**
     * Toggle between dark and light mode.
     */
    toggle() {
        if (this.isDarkMode()) {
            this.enableLightMode();
        } else {
            this.enableDarkMode();
        }
    }

    /**
     * Update the toggle button icon to reflect current theme.
     *
     * @private
     */
    updateToggleIcon() {
        if (!this.toggleBtn) return;

        const icon = this.toggleBtn.querySelector('i');
        if (!icon) return;

        if (this.isDarkMode()) {
            icon.className = 'fas fa-sun';
            this.toggleBtn.setAttribute('aria-label', 'Switch to light mode');
            this.toggleBtn.setAttribute('title', 'Switch to light mode');
        } else {
            icon.className = 'fas fa-moon';
            this.toggleBtn.setAttribute('aria-label', 'Switch to dark mode');
            this.toggleBtn.setAttribute('title', 'Switch to dark mode');
        }
    }
}
