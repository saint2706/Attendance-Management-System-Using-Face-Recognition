module.exports = {
  ci: {
    collect: {
      // Start the Django server for testing
      startServerCommand: 'python manage.py runserver 8000',
      startServerReadyPattern: 'Starting development server',
      startServerReadyTimeout: 60000,
      // URLs to audit
      url: [
        'http://localhost:8000/',
        'http://localhost:8000/login/',
      ],
      numberOfRuns: 3, // Run Lighthouse 3 times and take median
      settings: {
        // Ensure we're testing in a realistic scenario
        preset: 'desktop',
        // Use headless Chrome for CI environments
        chromeFlags: '--no-sandbox --disable-gpu --headless',
        throttling: {
          rttMs: 40,
          throughputKbps: 10240,
          cpuSlowdownMultiplier: 1,
        },
      },
    },
    assert: {
      assertions: {
        // Target scores as specified in README.md
        'categories:accessibility': ['error', {minScore: 0.95}],
        'categories:best-practices': ['error', {minScore: 0.95}],
        'categories:performance': ['error', {minScore: 0.80}],
        'categories:seo': ['error', {minScore: 0.90}],
        
        // Additional important checks
        'categories:pwa': 'off', // Not a PWA requirement
        
        // Specific best practices
        'uses-http2': 'off', // Development server doesn't use HTTP/2
        'redirects-http': 'off', // Not applicable for development
      },
    },
    upload: {
      target: 'temporary-public-storage', // Store results temporarily
    },
  },
};
