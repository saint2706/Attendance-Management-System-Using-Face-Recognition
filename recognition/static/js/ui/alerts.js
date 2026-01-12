/**
 * Alert management module for auto-dismissing alerts.
 *
 * @module ui/alerts
 */

import { ALERT_CONFIG } from '../core/config.js';

/**
 * Manages dismissible alerts with auto-dismiss functionality.
 */
export class AlertManager {
    /**
     * Initialize alert management.
     */
    init() {
        const alerts = document.querySelectorAll('.alert-dismissible');

        alerts.forEach(alert => {
            this._setupCloseButton(alert);
            this._setupAutoDismiss(alert);
        });
    }

    /**
     * Set up close button functionality.
     *
     * @private
     * @param {HTMLElement} alert - Alert element
     */
    _setupCloseButton(alert) {
        const closeBtn = alert.querySelector('.alert-close, .btn-close');

        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.dismiss(alert);
            });
        }
    }

    /**
     * Set up auto-dismiss timer.
     *
     * @private
     * @param {HTMLElement} alert - Alert element
     */
    _setupAutoDismiss(alert) {
        // Skip auto-dismiss if explicitly disabled
        if (alert.getAttribute('data-auto-dismiss') === 'false') {
            return;
        }

        setTimeout(() => {
            if (alert.parentElement) {
                this.dismiss(alert);
            }
        }, ALERT_CONFIG.AUTO_DISMISS_MS);
    }

    /**
     * Dismiss an alert with fade animation.
     *
     * @param {HTMLElement} alert - Alert element to dismiss
     */
    dismiss(alert) {
        alert.style.opacity = '0';
        alert.style.transform = 'translateY(-10px)';

        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, ALERT_CONFIG.FADE_OUT_MS);
    }

    /**
     * Show an alert programmatically.
     *
     * @param {string} message - Alert message
     * @param {string} [type='info'] - Alert type (success, info, warning, danger)
     * @param {boolean} [autoDismiss=true] - Whether to auto-dismiss
     */
    show(message, type = 'info', autoDismiss = true) {
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible`;
        alert.setAttribute('role', 'alert');

        if (!autoDismiss) {
            alert.setAttribute('data-auto-dismiss', 'false');
        }

        // 🛡️ Sentinel: Prevent XSS by wrapping message in a separate element
        const messageSpan = document.createElement('span');
        messageSpan.textContent = message;
        alert.appendChild(messageSpan);

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'btn-close';
        closeBtn.setAttribute('aria-label', 'Close');
        alert.appendChild(closeBtn);

        // Find or create alerts container
        let container = document.querySelector('.alerts-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'alerts-container';
            document.body.insertBefore(container, document.body.firstChild);
        }

        container.appendChild(alert);

        // Initialize the new alert
        this._setupCloseButton(alert);
        if (autoDismiss) {
            this._setupAutoDismiss(alert);
        }
    }
}
