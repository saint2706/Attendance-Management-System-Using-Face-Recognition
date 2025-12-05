/**
 * Form enhancement module for validation and floating labels.
 * 
 * @module ui/forms
 */

/**
 * Enhances forms with validation feedback and floating label animations.
 */
export class FormEnhancer {
    /**
     * Initialize form enhancements.
     */
    init() {
        // Add floating labels animation
        this._initFloatingLabels();

        // Add form validation feedback
        this._initValidation();
    }

    /**
     * Initialize floating labels for form inputs.
     * 
     * @private
     */
    _initFloatingLabels() {
        const inputs = document.querySelectorAll('.form-control');

        inputs.forEach(input => {
            // Set initial state
            if (input.value) {
                input.classList.add('has-value');
            }

            // Update on blur
            input.addEventListener('blur', () => {
                if (input.value) {
                    input.classList.add('has-value');
                } else {
                    input.classList.remove('has-value');
                }
            });
        });
    }

    /**
     * Initialize HTML5 form validation.
     * 
     * @private
     */
    _initValidation() {
        const forms = document.querySelectorAll('form[data-validate="true"]');

        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
    }
}
