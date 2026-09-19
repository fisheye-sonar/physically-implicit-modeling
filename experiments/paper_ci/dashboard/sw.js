// Minimal service worker: makes the page installable; state.json is always fetched live.
self.addEventListener('install', e => self.skipWaiting());
self.addEventListener('activate', e => self.clients.claim());
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (u.pathname.endsWith('state.json') || u.pathname.endsWith('ledger.json')) return;   // network only
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
