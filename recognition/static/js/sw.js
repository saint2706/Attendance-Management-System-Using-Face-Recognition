const CACHE_NAME = 'attendance-shell-v1';
const ATTENDANCE_DB = 'attendance-offline';
const ATTENDANCE_STORE = 'attendance-queue';
const ATTENDANCE_ENDPOINTS = [
  '/mark_your_attendance',
  '/mark_your_attendance_out'
];

const SHELL_ASSETS = [
  '/',
  '/static/css/app.css',
  '/static/css/styles.css',
  '/static/js/ui.js',
  '/static/js/camera.js',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames
          .filter((name) => name.startsWith('attendance-shell-') && name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method === 'POST' && isAttendanceRequest(url)) {
    event.respondWith(handleAttendanceSubmission(request));
    return;
  }

  if (request.method === 'GET') {
    if (request.mode === 'navigate') {
      event.respondWith(
        fetch(request).catch(() => caches.match('/'))
      );
      return;
    }

    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(request)
          .then((response) => {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
            return response;
          })
          .catch(() => caches.match('/'));
      })
    );
  }
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'attendance-sync') {
    event.waitUntil(flushAttendanceQueue());
  }
});

self.addEventListener('message', (event) => {
  if (!event.data) {
    return;
  }
  if (event.data === 'flushAttendanceQueue' || event.data?.type === 'FLUSH_ATTENDANCE_QUEUE') {
    event.waitUntil(flushAttendanceQueue());
  }
});

function isAttendanceRequest(url) {
  return ATTENDANCE_ENDPOINTS.some((endpoint) => url.pathname.includes(endpoint));
}

async function handleAttendanceSubmission(request) {
  try {
    const networkResponse = await fetch(request.clone());
    return networkResponse;
  } catch (error) {
    await queueAttendanceRequest(request);
    broadcastToClients({ type: 'attendance-queued' });
    return new Response(
      JSON.stringify({ queued: true, message: 'Attendance request saved for retry when online.' }),
      {
        status: 202,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

function openAttendanceDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(ATTENDANCE_DB, 1);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(ATTENDANCE_STORE)) {
        db.createObjectStore(ATTENDANCE_STORE, { keyPath: 'id', autoIncrement: true });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function queueAttendanceRequest(request) {
  const db = await openAttendanceDb();
  const cloned = request.clone();
  const body = await cloned.text();
  const headers = {};
  cloned.headers.forEach((value, key) => {
    headers[key] = value;
  });

  await new Promise((resolve, reject) => {
    const tx = db.transaction(ATTENDANCE_STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);

    tx.objectStore(ATTENDANCE_STORE).add({
      url: cloned.url,
      method: cloned.method,
      headers,
      body,
      timestamp: Date.now()
    });
  });

  db.close();
}

async function flushAttendanceQueue() {
  const db = await openAttendanceDb();

  const entries = await new Promise((resolve, reject) => {
    const tx = db.transaction(ATTENDANCE_STORE, 'readonly');
    const request = tx.objectStore(ATTENDANCE_STORE).getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });

  for (const entry of entries) {
    try {
      const response = await fetch(entry.url, {
        method: entry.method,
        headers: entry.headers,
        body: entry.body,
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error(`Failed with status ${response.status}`);
      }

      await removeEntry(db, entry.id);
    } catch (error) {
      // Leave the entry for a future retry
      console.error('Retrying attendance submission later', error);
      return;
    }
  }

  broadcastToClients({ type: 'attendance-synced' });
  db.close();
}

function removeEntry(db, id) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATTENDANCE_STORE, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.objectStore(ATTENDANCE_STORE).delete(id);
  });
}

function broadcastToClients(message) {
  self.clients.matchAll({ includeUncontrolled: true, type: 'window' }).then((clients) => {
    clients.forEach((client) => client.postMessage(message));
  });
}
