/**
 * Tooltip management module.
 * 
 * @module ui/tooltips
 */

/**
 * Manages tooltips for elements with data-tooltip attribute.
 */
export class TooltipManager {
    /**
     * Initialize tooltip functionality.
     */
    init() {
        const elements = document.querySelectorAll('[data-tooltip]');

        elements.forEach(element => {
            element.addEventListener('mouseenter', (e) => this.show(e.target));
            element.addEventListener('mouseleave', (e) => this.hide(e.target));
            element.addEventListener('focus', (e) => this.show(e.target));
            element.addEventListener('blur', (e) => this.hide(e.target));
        });
    }

    /**
     * Show tooltip for an element.
     * 
     * @param {HTMLElement} element - Element to show tooltip for
     */
    show(element) {
        const text = element.getAttribute('data-tooltip');
        if (!text) return;

        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip-content';
        tooltip.textContent = text;
        tooltip.style.cssText = `
      position: absolute;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 0.5rem 0.75rem;
      border-radius: 0.25rem;
      font-size: 0.875rem;
      z-index: 9999;
      pointer-events: none;
    `;

        document.body.appendChild(tooltip);

        // Position the tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';

        // Store reference for cleanup
        element._tooltip = tooltip;
    }

    /**
     * Hide tooltip for an element.
     * 
     * @param {HTMLElement} element - Element to hide tooltip for
     */
    hide(element) {
        if (element._tooltip) {
            element._tooltip.remove();
            delete element._tooltip;
        }
    }

    /**
     * Hide all tooltips.
     */
    hideAll() {
        document.querySelectorAll('.tooltip-content').forEach(tooltip => {
            tooltip.remove();
        });
    }
}
