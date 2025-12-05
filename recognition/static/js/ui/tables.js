/**
 * Table enhancements module for search, sort, and CSV export.
 * 
 * @module ui/tables
 */

import { TABLE_CONFIG } from '../core/config.js';

/**
 * Enhances HTML tables with search filtering, sorting, and CSV export capabilities.
 */
export class TableEnhancer {
    /**
     * Initialize table enhancements for all tables with data-enhance="true".
     */
    init() {
        const tables = document.querySelectorAll('table[data-enhance="true"]');
        tables.forEach(table => this.enhanceTable(table));
    }

    /**
     * Enhance a single table with all features.
     * 
     * @param {HTMLTableElement} table - The table element to enhance
     */
    enhanceTable(table) {
        const container = table.closest('.table-container') || table.parentElement;

        // Add search/filter
        this.addSearchFilter(table, container);

        // Add CSV export button
        this.addExportButton(table, container);

        // Add sortable columns
        this.makeSortable(table);
    }

    /**
     * Add search filter input above table.
     * 
     * @param {HTMLTableElement} table - The table to add search to
     * @param {HTMLElement} container - Container element for controls
     */
    addSearchFilter(table, container) {
        // Check if search already exists
        if (container.querySelector('.table-search')) return;

        const controls = document.createElement('div');
        controls.className = 'table-controls';

        const searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.className = 'form-control table-search';
        searchInput.placeholder = 'Search table...';
        searchInput.setAttribute('aria-label', 'Search table');

        controls.appendChild(searchInput);

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'table-actions';
        controls.appendChild(actionsDiv);

        container.insertBefore(controls, table);

        // Implement search functionality
        searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            const tbody = table.querySelector('tbody');
            if (!tbody) return;

            const rows = tbody.querySelectorAll('tr');
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        });
    }

    /**
     * Add CSV export button to table controls.
     * 
     * @param {HTMLTableElement} table - The table to export
     * @param {HTMLElement} container - Container element for controls
     */
    addExportButton(table, container) {
        const controls = container.querySelector('.table-controls');
        if (!controls) return;

        const actionsDiv = controls.querySelector('.table-actions');
        if (!actionsDiv) return;

        const exportBtn = document.createElement('button');
        exportBtn.className = 'btn btn-outline-primary';
        exportBtn.innerHTML = '<i class="fas fa-download"></i> Export CSV';
        exportBtn.setAttribute('aria-label', 'Export table as CSV');

        exportBtn.addEventListener('click', () => this.exportToCSV(table));

        actionsDiv.appendChild(exportBtn);
    }

    /**
     * Export table data to CSV file.
     * 
     * @param {HTMLTableElement} table - The table to export
     */
    exportToCSV(table) {
        const rows = [];

        // Get headers
        const headers = [];
        table.querySelectorAll('thead th').forEach(th => {
            headers.push(this._escapeCSV(th.textContent.trim()));
        });
        rows.push(headers.join(TABLE_CONFIG.CSV_DELIMITER));

        // Get data rows
        table.querySelectorAll('tbody tr').forEach(tr => {
            // Skip hidden rows (filtered out)
            if (tr.style.display === 'none') return;

            const cells = [];
            tr.querySelectorAll('td').forEach(td => {
                cells.push(this._escapeCSV(td.textContent.trim()));
            });
            rows.push(cells.join(TABLE_CONFIG.CSV_DELIMITER));
        });

        // Create and download file
        const csv = rows.join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${TABLE_CONFIG.CSV_FILENAME_PREFIX}_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Escape CSV special characters.
     * 
     * @private
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    _escapeCSV(str) {
        // Escape quotes and wrap in quotes if contains comma, quote, or newline
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    }

    /**
     * Make table columns sortable by clicking headers.
     * 
     * @param {HTMLTableElement} table - The table to make sortable
     */
    makeSortable(table) {
        const headers = table.querySelectorAll('thead th');

        headers.forEach((header, index) => {
            // Skip if header has data-sortable="false"
            if (header.getAttribute('data-sortable') === 'false') return;

            header.style.cursor = 'pointer';
            header.setAttribute('role', 'button');
            header.setAttribute('tabindex', '0');

            const originalContent = header.innerHTML;
            header.innerHTML = originalContent + ' <i class="fas fa-sort" style="opacity: 0.3; margin-left: 0.25rem;"></i>';

            const sortIcon = header.querySelector('i');

            header.addEventListener('click', () => this._sortTable(table, index, header, sortIcon));
            header.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this._sortTable(table, index, header, sortIcon);
                }
            });
        });
    }

    /**
     * Sort table by column.
     * 
     * @private
     * @param {HTMLTableElement} table - The table to sort
     * @param {number} columnIndex - Column index to sort by
     * @param {HTMLElement} header - Header element
     * @param {HTMLElement} icon - Sort icon element
     */
    _sortTable(table, columnIndex, header, icon) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));
        const currentOrder = header.getAttribute('data-sort-order') || 'none';

        // Determine new order
        let newOrder = 'asc';
        if (currentOrder === 'asc') {
            newOrder = 'desc';
        }

        // Reset all other headers
        table.querySelectorAll('thead th').forEach(th => {
            th.removeAttribute('data-sort-order');
            const thIcon = th.querySelector('i.fa-sort, i.fa-sort-up, i.fa-sort-down');
            if (thIcon && thIcon !== icon) {
                thIcon.className = 'fas fa-sort';
                thIcon.style.opacity = '0.3';
            }
        });

        // Update current header
        header.setAttribute('data-sort-order', newOrder);
        icon.className = newOrder === 'asc' ? 'fas fa-sort-up' : 'fas fa-sort-down';
        icon.style.opacity = '1';

        // Sort rows
        rows.sort((a, b) => {
            const aCell = a.querySelectorAll('td')[columnIndex];
            const bCell = b.querySelectorAll('td')[columnIndex];

            if (!aCell || !bCell) return 0;

            const aText = aCell.textContent.trim();
            const bText = bCell.textContent.trim();

            // Try to parse as numbers
            const aNum = parseFloat(aText);
            const bNum = parseFloat(bText);

            let comparison = 0;
            if (!isNaN(aNum) && !isNaN(bNum)) {
                comparison = aNum - bNum;
            } else {
                comparison = aText.localeCompare(bText);
            }

            return newOrder === 'asc' ? comparison : -comparison;
        });

        // Re-append rows in new order
        rows.forEach(row => tbody.appendChild(row));
    }
}
