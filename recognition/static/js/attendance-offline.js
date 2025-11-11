(function() {
  'use strict';

  if (!('fetch' in window) || !('indexedDB' in window)) {
    return;
  }

  const originalFetch = window.fetch.bind(window);
  const DB_NAME = 'attendance-offline';
  const STORE_NAME = 'attendance-queue';
  const ATTENDANCE_ENDPOINTS = [
    '/mark_your_attendance',
    '/mark_your_attendance_out'
  ];

  let dbPromise = null;

  function openDb() {
    if (dbPromise) {
      return dbPromise;
    }

    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);

      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains(STORE_NAME)) {
          database.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    return dbPromise;
  }

  function isAttendanceUrl(input) {
    try {
      const url = new URL(input, window.location.origin);
      return ATTENDANCE_ENDPOINTS.some((endpoint) => url.pathname.includes(endpoint));
    } catch (error) {
      return false;
    }
  }

  async function saveRequest(request) {
    const database = await openDb();
    const clone = request.clone();
    const headers = {};
    clone.headers.forEach((value, key) => {
      headers[key] = value;
    });

    const body = clone.method === 'GET' || clone.method === 'HEAD' ? null : await clone.text();

    await new Promise((resolve, reject) => {
      const tx = database.transaction(STORE_NAME, 'readwrite');
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.objectStore(STORE_NAME).add({
        url: clone.url,
        method: clone.method,
        headers,
        body,
        timestamp: Date.now()
      });
    });

    document.dispatchEvent(new CustomEvent('attendance-queued'));
  }

  async function flushQueue() {
    try {
      const database = await openDb();
      const entries = await new Promise((resolve, reject) => {
        const tx = database.transaction(STORE_NAME, 'readonly');
        const request = tx.objectStore(STORE_NAME).getAll();
        request.onsuccess = () => resolve(request.result || []);
        request.onerror = () => reject(request.error);
      });

      for (const entry of entries) {
        try {
          const response = await originalFetch(entry.url, {
            method: entry.method,
            headers: entry.headers,
            body: entry.body,
            credentials: 'include'
          });

          if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
          }

          await new Promise((resolve, reject) => {
            const tx = database.transaction(STORE_NAME, 'readwrite');
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
            tx.objectStore(STORE_NAME).delete(entry.id);
          });
        } catch (error) {
          console.warn('Attendance submission retry failed; will try again later.', error);
          return;
        }
      }

      document.dispatchEvent(new CustomEvent('attendance-synced'));
    } catch (error) {
      console.error('Unable to flush attendance queue', error);
    }
  }

  async function offlineFetch(input, init) {
    if (!isAttendanceUrl(typeof input === 'string' ? input : input.url)) {
      return originalFetch(input, init);
    }

    if (navigator.onLine) {
      return originalFetch(input, init);
    }

    const request = new Request(input, init);
    await saveRequest(request);

    if (navigator.serviceWorker && navigator.serviceWorker.controller) {
      navigator.serviceWorker.controller.postMessage({ type: 'FLUSH_ATTENDANCE_QUEUE' });
    }

    return new Response(
      JSON.stringify({ queued: true, message: 'You appear to be offline. Attendance will be submitted automatically.' }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }

  window.fetch = function(input, init) {
    return offlineFetch(input, init);
  };

  window.addEventListener('online', () => {
    flushQueue();
    if (navigator.serviceWorker?.ready) {
      navigator.serviceWorker.ready.then((registration) => {
        if ('sync' in registration) {
          registration.sync.register('attendance-sync').catch(() => {
            navigator.serviceWorker.controller?.postMessage({ type: 'FLUSH_ATTENDANCE_QUEUE' });
          });
        } else {
          navigator.serviceWorker.controller?.postMessage({ type: 'FLUSH_ATTENDANCE_QUEUE' });
        }
      });
    }
  });

  navigator.serviceWorker?.addEventListener('message', (event) => {
    if (event.data?.type === 'attendance-synced') {
      flushQueue();
    }
  });

  window.AttendanceOffline = {
    flushQueue,
    enqueue: async (url, options) => {
      const request = new Request(url, options);
      await saveRequest(request);
      if (navigator.serviceWorker?.ready) {
        navigator.serviceWorker.ready.then((registration) => {
          if ('sync' in registration) {
            registration.sync.register('attendance-sync').catch(() => {
              navigator.serviceWorker.controller?.postMessage({ type: 'FLUSH_ATTENDANCE_QUEUE' });
            });
          }
        });
      }
    }
  };

  if (!navigator.onLine) {
    // Attempt to resubmit any entries created before the current page load when the network returns.
    document.addEventListener('visibilitychange', () => {
      if (navigator.onLine && document.visibilityState === 'visible') {
        flushQueue();
      }
    });
  } else {
    flushQueue();
  }
})();
