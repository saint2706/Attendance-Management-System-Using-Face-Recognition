/**
 * Smart Attendance System - Main Application Bootstrap
 * 
 * This is the entry point for all frontend JavaScript modules.
 * It initializes core utilities and UI enhancements in the correct order.
 * 
 * @module main
 */

// Import core utilities
import { globalBus } from './core/events.js';
import { THEME_CONFIG } from './core/config.js';

// Import UI modules
import { ThemeManager } from './ui/theme.js';
import { MobileNav } from './ui/navigation.js';
import { TableEnhancer } from './ui/tables.js';
import { FormEnhancer } from './ui/forms.js';
import { AlertManager } from './ui/alerts.js';
import { TooltipManager } from './ui/tooltips.js';

/**
 * Application class that manages all modules.
 */
class AttendanceApp {
    constructor() {
        this.modules = {};
        this.initialized = false;
    }

    /**
     * Initialize all application modules.
     */
    init() {
        if (this.initialized) {
            console.warn('[AttendanceApp] Already initialized');
            return;
        }

        try {
            // Initialize modules in dependency order
            this._initCore();
            this._initUI();
            this._setupEventListeners();

            this.initialized = true;
            console.log('[AttendanceApp] ✓ All modules initialized successfully');

            // Emit initialization complete event
            globalBus.emit('app:ready', { modules: Object.keys(this.modules) });

        } catch (error) {
            console.error('[AttendanceApp] Initialization error:', error);
            this._handleInitError(error);
        }
    }

    /**
     * Initialize core modules.
     * 
     * @private
     */
    _initCore() {
        // EventBus is already initialized globally
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            // Enable debug logging in development
            // globalBus.debug = true;
        }
    }

    /**
     * Initialize UI modules.
     * 
     * @private
     */
    _initUI() {
        // Theme management
        this.modules.theme = new ThemeManager();
        this.modules.theme.init();

        // Mobile navigation
        this.modules.navigation = new MobileNav();
        this.modules.navigation.init();

        // Table enhancements
        this.modules.tables = new TableEnhancer();
        this.modules.tables.init();

        // Form enhancements
        this.modules.forms = new FormEnhancer();
        this.modules.forms.init();

        // Alert management
        this.modules.alerts = new AlertManager();
        this.modules.alerts.init();

        // Tooltips
        this.modules.tooltips = new TooltipManager();
        this.modules.tooltips.init();

        // Card animations (inline for now, could be extracted later)
        this._initCardAnimations();
    }

    /**
     * Initialize card animations with Intersection Observer.
     * 
     * @private
     */
    _initCardAnimations() {
        const cards = document.querySelectorAll('.card');

        if (cards.length === 0) return;

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, {
            threshold: 0.1
        });

        cards.forEach(card => observer.observe(card));
    }

    /**
     * Set up global event listeners.
     * 
     * @private
     */
    _setupEventListeners() {
        // Example: Listen to theme changes
        globalBus.on('theme:changed', (data) => {
            console.log('[AttendanceApp] Theme changed to:', data.theme);
        });

        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            this.dispose();
        });
    }

    /**
     * Handle initialization errors gracefully.
     * 
     * @private
     * @param {Error} error - The error that occurred
     */
    _handleInitError(error) {
        // Show user-friendly error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger';
        errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000;';
        errorDiv.textContent = 'Application initialization failed. Please refresh the page.';

        document.body.appendChild(errorDiv);

        // Auto-remove after 10 seconds
        setTimeout(() => errorDiv.remove(), 10000);
    }

    /**
     * Get a specific module instance.
     * 
     * @param {string} name - Module name
     * @returns {Object|null} Module instance or null
     */
    getModule(name) {
        return this.modules[name] || null;
    }

    /**
     * Clean up resources on page unload.
     */
    dispose() {
        console.log('[AttendanceApp] Cleaning up...');
        globalBus.clear();
        this.modules = {};
        this.initialized = false;
    }
}

// Create global app instance
const app = new AttendanceApp();

/**
 * Initialize the application when DOM is ready.
 */
function bootstrap() {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => app.init());
    } else {
        app.init();
    }
}

// Run bootstrap
bootstrap();

// Expose app instance for debugging and third-party integrations
window.AttendanceApp = app;

// Export for testing
export { AttendanceApp, app };
