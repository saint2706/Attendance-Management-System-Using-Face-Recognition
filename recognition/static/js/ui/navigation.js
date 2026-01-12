/**
 * Mobile navigation menu module.
 *
 * @module ui/navigation
 */

/**
 * Manages the mobile navigation menu toggle, outside clicks, and keyboard navigation.
 */
export class MobileNav {
    constructor() {
        this.toggleBtn = null;
        this.nav = null;
    }

    /**
     * Initialize mobile navigation functionality.
     */
    init() {
        this.toggleBtn = document.getElementById('mobile-menu-toggle');
        this.nav = document.getElementById('navbar-nav');

        if (!this.toggleBtn || !this.nav) return;

        this._setupToggle();
        this._setupOutsideClick();
        this._setupKeyboardNav();
    }

    /**
     * Set up the toggle button click handler.
     *
     * @private
     */
    _setupToggle() {
        this.toggleBtn.addEventListener('click', () => {
            const isOpen = this.nav.classList.toggle('is-open');
            this.toggleBtn.setAttribute('aria-expanded', isOpen);

            // Update icon
            const icon = this.toggleBtn.querySelector('i');
            if (icon) {
                icon.className = isOpen ? 'fas fa-times' : 'fas fa-bars';
            }
        });
    }

    /**
     * Close menu when clicking outside.
     *
     * @private
     */
    _setupOutsideClick() {
        document.addEventListener('click', (e) => {
            if (!this.toggleBtn.contains(e.target) && !this.nav.contains(e.target)) {
                this.close();
            }
        });
    }

    /**
     * Close menu when pressing Escape key.
     *
     * @private
     */
    _setupKeyboardNav() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.nav.classList.contains('is-open')) {
                this.close();
                this.toggleBtn.focus();
            }
        });
    }

    /**
     * Close the mobile navigation menu.
     */
    close() {
        this.nav.classList.remove('is-open');
        this.toggleBtn.setAttribute('aria-expanded', 'false');

        const icon = this.toggleBtn.querySelector('i');
        if (icon) {
            icon.className = 'fas fa-bars';
        }
    }

    /**
     * Open the mobile navigation menu.
     */
    open() {
        this.nav.classList.add('is-open');
        this.toggleBtn.setAttribute('aria-expanded', 'true');

        const icon = this.toggleBtn.querySelector('i');
        if (icon) {
            icon.className = 'fas fa-times';
        }
    }

    /**
     * Toggle the mobile navigation menu.
     */
    toggle() {
        if (this.nav.classList.contains('is-open')) {
            this.close();
        } else {
            this.open();
        }
    }
}
