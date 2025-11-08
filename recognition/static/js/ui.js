/**
 * Smart Attendance System - UI JavaScript
 * 
 * Handles dark mode toggle, mobile navigation, table filtering,
 * CSV export, and other UI enhancements.
 */

(function() {
  'use strict';

  // ============================================================================
  // THEME MANAGEMENT
  // ============================================================================
  
  const ThemeManager = {
    STORAGE_KEY: 'attendance-theme',
    DARK_CLASS: 'theme-dark',
    
    init() {
      // Load saved theme preference or default to light
      const savedTheme = this.getSavedTheme();
      if (savedTheme === 'dark') {
        this.enableDarkMode();
      }
      
      // Set up theme toggle button
      const toggleBtn = document.getElementById('theme-toggle');
      if (toggleBtn) {
        toggleBtn.addEventListener('click', () => this.toggle());
        this.updateToggleIcon();
      }
    },
    
    getSavedTheme() {
      return localStorage.getItem(this.STORAGE_KEY);
    },
    
    saveTheme(theme) {
      localStorage.setItem(this.STORAGE_KEY, theme);
    },
    
    isDarkMode() {
      return document.documentElement.classList.contains(this.DARK_CLASS);
    },
    
    enableDarkMode() {
      document.documentElement.classList.add(this.DARK_CLASS);
      this.saveTheme('dark');
      this.updateToggleIcon();
    },
    
    enableLightMode() {
      document.documentElement.classList.remove(this.DARK_CLASS);
      this.saveTheme('light');
      this.updateToggleIcon();
    },
    
    toggle() {
      if (this.isDarkMode()) {
        this.enableLightMode();
      } else {
        this.enableDarkMode();
      }
    },
    
    updateToggleIcon() {
      const toggleBtn = document.getElementById('theme-toggle');
      if (!toggleBtn) return;
      
      const icon = toggleBtn.querySelector('i');
      if (!icon) return;
      
      if (this.isDarkMode()) {
        icon.className = 'fas fa-sun';
        toggleBtn.setAttribute('aria-label', 'Switch to light mode');
        toggleBtn.setAttribute('title', 'Switch to light mode');
      } else {
        icon.className = 'fas fa-moon';
        toggleBtn.setAttribute('aria-label', 'Switch to dark mode');
        toggleBtn.setAttribute('title', 'Switch to dark mode');
      }
    }
  };

  // ============================================================================
  // MOBILE NAVIGATION
  // ============================================================================
  
  const MobileNav = {
    init() {
      const toggleBtn = document.getElementById('mobile-menu-toggle');
      const nav = document.getElementById('navbar-nav');
      
      if (!toggleBtn || !nav) return;
      
      toggleBtn.addEventListener('click', () => {
        const isOpen = nav.classList.toggle('is-open');
        toggleBtn.setAttribute('aria-expanded', isOpen);
        
        // Update icon
        const icon = toggleBtn.querySelector('i');
        if (icon) {
          icon.className = isOpen ? 'fas fa-times' : 'fas fa-bars';
        }
      });
      
      // Close menu when clicking outside
      document.addEventListener('click', (e) => {
        if (!toggleBtn.contains(e.target) && !nav.contains(e.target)) {
          nav.classList.remove('is-open');
          toggleBtn.setAttribute('aria-expanded', 'false');
          const icon = toggleBtn.querySelector('i');
          if (icon) {
            icon.className = 'fas fa-bars';
          }
        }
      });
      
      // Close menu when pressing Escape
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && nav.classList.contains('is-open')) {
          nav.classList.remove('is-open');
          toggleBtn.setAttribute('aria-expanded', 'false');
          toggleBtn.focus();
        }
      });
    }
  };

  // ============================================================================
  // TABLE ENHANCEMENTS
  // ============================================================================
  
  const TableEnhancer = {
    init() {
      // Find all tables with data-enhance="true"
      const tables = document.querySelectorAll('table[data-enhance="true"]');
      tables.forEach(table => this.enhanceTable(table));
    },
    
    enhanceTable(table) {
      const container = table.closest('.table-container') || table.parentElement;
      
      // Add search/filter
      this.addSearchFilter(table, container);
      
      // Add CSV export button
      this.addExportButton(table, container);
      
      // Add sortable columns
      this.makeSortable(table);
    },
    
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
    },
    
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
    },
    
    exportToCSV(table) {
      const rows = [];
      
      // Get headers
      const headers = [];
      table.querySelectorAll('thead th').forEach(th => {
        headers.push(this.escapeCSV(th.textContent.trim()));
      });
      rows.push(headers.join(','));
      
      // Get data rows
      table.querySelectorAll('tbody tr').forEach(tr => {
        // Skip hidden rows (filtered out)
        if (tr.style.display === 'none') return;
        
        const cells = [];
        tr.querySelectorAll('td').forEach(td => {
          cells.push(this.escapeCSV(td.textContent.trim()));
        });
        rows.push(cells.join(','));
      });
      
      // Create and download file
      const csv = rows.join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `table-export-${Date.now()}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    },
    
    escapeCSV(str) {
      // Escape quotes and wrap in quotes if contains comma, quote, or newline
      if (str.includes(',') || str.includes('"') || str.includes('\n')) {
        return '"' + str.replace(/"/g, '""') + '"';
      }
      return str;
    },
    
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
        
        header.addEventListener('click', () => this.sortTable(table, index, header, sortIcon));
        header.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.sortTable(table, index, header, sortIcon);
          }
        });
      });
    },
    
    sortTable(table, columnIndex, header, icon) {
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
  };

  // ============================================================================
  // FORM ENHANCEMENTS
  // ============================================================================
  
  const FormEnhancer = {
    init() {
      // Add floating labels animation
      this.initFloatingLabels();
      
      // Add form validation feedback
      this.initValidation();
    },
    
    initFloatingLabels() {
      const inputs = document.querySelectorAll('.form-control');
      inputs.forEach(input => {
        if (input.value) {
          input.classList.add('has-value');
        }
        
        input.addEventListener('blur', () => {
          if (input.value) {
            input.classList.add('has-value');
          } else {
            input.classList.remove('has-value');
          }
        });
      });
    },
    
    initValidation() {
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
  };

  // ============================================================================
  // ALERT AUTO-DISMISS
  // ============================================================================
  
  const AlertManager = {
    init() {
      const alerts = document.querySelectorAll('.alert-dismissible');
      
      alerts.forEach(alert => {
        const closeBtn = alert.querySelector('.alert-close, .btn-close');
        if (closeBtn) {
          closeBtn.addEventListener('click', () => {
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-10px)';
            setTimeout(() => alert.remove(), 300);
          });
        }
        
        // Auto-dismiss after 5 seconds
        if (alert.getAttribute('data-auto-dismiss') !== 'false') {
          setTimeout(() => {
            if (alert.parentElement) {
              alert.style.opacity = '0';
              alert.style.transform = 'translateY(-10px)';
              setTimeout(() => alert.remove(), 300);
            }
          }, 5000);
        }
      });
    }
  };

  // ============================================================================
  // TOOLTIPS
  // ============================================================================
  
  const TooltipManager = {
    init() {
      const elements = document.querySelectorAll('[data-tooltip]');
      
      elements.forEach(element => {
        element.addEventListener('mouseenter', (e) => this.show(e.target));
        element.addEventListener('mouseleave', (e) => this.hide(e.target));
        element.addEventListener('focus', (e) => this.show(e.target));
        element.addEventListener('blur', (e) => this.hide(e.target));
      });
    },
    
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
      
      const rect = element.getBoundingClientRect();
      tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
      tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
      
      element._tooltip = tooltip;
    },
    
    hide(element) {
      if (element._tooltip) {
        element._tooltip.remove();
        delete element._tooltip;
      }
    }
  };

  // ============================================================================
  // CARD ANIMATIONS
  // ============================================================================
  
  const CardAnimator = {
    init() {
      const cards = document.querySelectorAll('.card');
      
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
          }
        });
      }, {
        threshold: 0.1
      });
      
      cards.forEach(card => {
        observer.observe(card);
      });
    }
  };

  // ============================================================================
  // INITIALIZATION
  // ============================================================================
  
  function init() {
    // Initialize all modules
    ThemeManager.init();
    MobileNav.init();
    TableEnhancer.init();
    FormEnhancer.init();
    AlertManager.init();
    TooltipManager.init();
    CardAnimator.init();
    
    console.log('UI enhancements initialized');
  }
  
  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export for testing
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      ThemeManager,
      MobileNav,
      TableEnhancer,
      FormEnhancer,
      AlertManager
    };
  }
})();
