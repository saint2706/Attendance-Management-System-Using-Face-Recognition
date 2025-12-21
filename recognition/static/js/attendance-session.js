/**
 * Attendance Session Monitor - Live attendance log updates.
 * 
 * @module attendance-session
 */

import { POLLING_CONFIG } from './core/config.js';

/**
 * Monitors attendance session feed and updates the UI in real-time.
 */
export class AttendanceSessionMonitor {
    /**
     * @param {HTMLElement} logContainer - Container element with data-feed-url attribute
     */
    constructor(logContainer) {
        this.logContainer = logContainer;
        this.tbody = document.getElementById('attendance-log-body');
        this.feedUrl = logContainer?.dataset?.feedUrl;
        this.pollInterval = null;
        this.errorCount = 0;
    }

    /**
     * Start monitoring the attendance feed.
     */
    start() {
        if (!this.logContainer || !this.tbody || !this.feedUrl) {
            console.warn('[AttendanceSession] Missing required elements or feed URL');
            return;
        }

        // Initial fetch
        this._fetchFeed();

        // Poll for updates
        this.pollInterval = setInterval(
            () => this._fetchFeed(),
            POLLING_CONFIG.ATTENDANCE_LOG_INTERVAL_MS
        );

        // Clean up on page unload
        window.addEventListener('beforeunload', () => this.stop());

        // Pause polling when page is hidden
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pause();
            } else {
                this.resume();
            }
        });
    }

    /**
     * Stop monitoring and clean up resources.
     */
    stop() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    /**
     * Pause polling (useful when tab is in background).
     */
    pause() {
        this.stop();
    }

    /**
     * Resume polling after pause.
     */
    resume() {
        if (!this.pollInterval) {
            this.start();
        }
    }

    /**
     * Fetch attendance feed from server.
     * 
     * @private
     */
    async _fetchFeed() {
        try {
            const response = await fetch(this.feedUrl, { credentials: 'same-origin' });

            if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
            }

            const payload = await response.json();
            this._renderRows(payload.events || []);

            // Reset error count on success
            this.errorCount = 0;

        } catch (error) {
            this.errorCount++;

            console.error('[AttendanceSession] Fetch error:', error);

            this.tbody.innerHTML = `<tr><td colspan="6" class="text-center text-danger py-4">Unable to load live log (${this._escapeHtml(error.message)}).</td></tr>`;

            // Stop polling after too many errors
            if (this.errorCount >= POLLING_CONFIG.MAX_POLLING_ERRORS) {
                console.error('[AttendanceSession] Too many errors, stopping polling');
                this.stop();
            }
        }
    }

    /**
     * Render attendance events as table rows.
     * 
     * @private
     * @param {Array} events - Array of attendance events
     */
    _renderRows(events) {
        if (!events || events.length === 0) {
            this.tbody.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">No recent recognition events.</td></tr>';
            return;
        }

        const rows = events.map((event) => this._renderEvent(event));
        this.tbody.innerHTML = rows.join('');
    }

    /**
     * Render a single attendance event.
     * 
     * @private
     * @param {Object} event - Attendance event object
     * @returns {string} HTML string for table row
     */
    _renderEvent(event) {
        const timestamp = new Date(event.timestamp).toLocaleString();
        // 🛡️ Sentinel: Escape user input to prevent XSS
        const username = this._escapeHtml(event.username || 'Unknown');
        const direction = this._escapeHtml(event.direction || '—');

        let status = 'Pending';
        let statusStyle = 'bg-secondary';
        let liveness = 'Not checked';
        let livenessStyle = 'bg-secondary';
        let confidence = '—';

        if (event.event_type === 'outcome') {
            status = event.accepted ? 'Accepted' : 'Rejected';
            statusStyle = event.accepted ? 'bg-success' : 'bg-danger';

            if (event.confidence !== null && event.confidence !== undefined) {
                confidence = `${(event.confidence * 100).toFixed(1)}%`;
            } else if (event.distance !== null && event.threshold !== null) {
                confidence = `dist ${event.distance.toFixed(3)} / ${event.threshold.toFixed(3)}`;
            }
        } else {
            status = event.successful ? 'Recognized' : 'Attempted';
            statusStyle = event.successful ? 'bg-success' : 'bg-secondary';

            if (event.liveness === 'failed') {
                liveness = 'Failed';
                livenessStyle = 'bg-warning text-dark';
            } else if (event.liveness === 'passed') {
                liveness = 'Passed';
                livenessStyle = 'bg-success';
            }

            if (event.error) {
                status = 'Error';
                statusStyle = 'bg-danger';
                // 🛡️ Sentinel: Escape error message too
                confidence = this._escapeHtml(event.error);
            }
        }

        return `
      <tr>
        <td class="text-nowrap">${timestamp}</td>
        <td>${username}</td>
        <td class="text-capitalize">${direction}</td>
        <td>${this._statusBadge(status, statusStyle)}</td>
        <td>${this._statusBadge(liveness, livenessStyle)}</td>
        <td>${confidence}</td>
      </tr>
    `;
    }

    /**
     * Create a status badge HTML.
     * 
     * @private
     * @param {string} label - Badge label
     * @param {string} style - Badge CSS class
     * @returns {string} HTML string for badge
     */
    _statusBadge(label, style) {
        return `<span class="badge ${style}">${label}</span>`;
    }

    /**
     * Escape HTML characters to prevent XSS.
     *
     * @private
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    _escapeHtml(text) {
        if (!text) return text;
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}
