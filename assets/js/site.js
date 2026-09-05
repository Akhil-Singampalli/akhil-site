/* Shared chrome for every page: theme, nav, reveal, scroll-spy, lightbox.
   Every block guards on the elements it needs, so pages can include only
   the parts of the layout they actually use. */
(function () {
  'use strict';

  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var root = document.documentElement;

  /* ---------- Theme toggle ---------- */
  function currentTheme() {
    var set = root.getAttribute('data-theme');
    if (set) return set;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    var syncToggleLabel = function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      toggle.setAttribute('aria-label', 'Switch to ' + next + ' theme');
    };
    toggle.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      syncToggleLabel();
      /* Canvases paint from CSS tokens, so they have to be told to repaint. */
      window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
    });
    syncToggleLabel();
  }

  /* A page with no explicit choice still follows the OS. */
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
    if (!root.getAttribute('data-theme')) {
      window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: currentTheme() } }));
    }
  });

  /* ---------- Mobile menu ---------- */
  var navToggle = document.getElementById('nav-toggle');
  var navMenu = document.getElementById('nav-menu');

  function closeMenu() {
    if (!navMenu || !navToggle) return;
    navMenu.classList.remove('is-open');
    navToggle.setAttribute('aria-expanded', 'false');
    navToggle.setAttribute('aria-label', 'Open menu');
  }

  if (navToggle && navMenu) {
    navToggle.addEventListener('click', function () {
      var open = navMenu.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', String(open));
      navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
    navMenu.addEventListener('click', function (e) {
      if (e.target.closest('a')) closeMenu();
    });
  }

  /* ---------- Sticky nav shadow + back to top ---------- */
  var nav = document.getElementById('nav');
  var toTop = document.getElementById('to-top');

  if (nav || toTop) {
    var onScroll = function () {
      var y = window.scrollY;
      if (nav) nav.classList.toggle('is-stuck', y > 8);
      if (toTop) toTop.classList.toggle('is-visible', y > 700);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  if (toTop) {
    toTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' });
    });
  }

  /* ---------- Scroll reveal ---------- */
  var revealables = document.querySelectorAll('.reveal');
  if (reduced || !('IntersectionObserver' in window)) {
    revealables.forEach(function (el) { el.classList.add('is-in'); });
  } else {
    var revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-in');
        revealObserver.unobserve(entry.target);
      });
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.08 });
    revealables.forEach(function (el) { revealObserver.observe(el); });
  }

  /* ---------- Scroll spy ----------
     Used by the home nav and by the explorer page's section rail. */
  function spyOn(anchors) {
    var list = Array.prototype.slice.call(anchors);
    var sections = list
      .map(function (a) {
        var href = a.getAttribute('href') || '';
        return href.charAt(0) === '#' && href.length > 1 ? document.querySelector(href) : null;
      })
      .filter(Boolean);

    if (!('IntersectionObserver' in window) || !sections.length) return;

    var visible = new Map();
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        visible.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
      });
      var bestId = null, bestRatio = 0;
      visible.forEach(function (ratio, id) {
        if (ratio > bestRatio) { bestRatio = ratio; bestId = id; }
      });
      list.forEach(function (a) {
        a.classList.toggle('is-active', bestId !== null && a.getAttribute('href') === '#' + bestId);
      });
    }, { rootMargin: '-25% 0px -55% 0px', threshold: [0, 0.15, 0.4, 0.75, 1] });

    sections.forEach(function (s) { spy.observe(s); });
  }

  if (navMenu) spyOn(navMenu.querySelectorAll('a'));
  var rail = document.querySelector('[data-spy]');
  if (rail) spyOn(rail.querySelectorAll('a'));

  /* ---------- Figure lightbox ---------- */
  var lightbox = document.getElementById('lightbox');
  var lightboxImg = document.getElementById('lightbox-img');
  var lightboxCap = document.getElementById('lightbox-cap');
  var lightboxClose = document.getElementById('lightbox-close');
  var lastFocused = null;

  function closeLightbox() {
    lightbox.classList.remove('is-open');
    document.body.style.overflow = '';
    window.setTimeout(function () {
      lightbox.hidden = true;
      lightboxImg.removeAttribute('src');
    }, reduced ? 0 : 250);
    if (lastFocused) lastFocused.focus();
  }

  if (lightbox && lightboxImg && lightboxCap && lightboxClose) {
    var openLightbox = function (trigger) {
      lastFocused = trigger;
      var img = trigger.querySelector('img');
      lightboxImg.src = trigger.dataset.full || img.src;
      lightboxImg.alt = img.alt;
      lightboxCap.innerHTML = '';

      var title = document.createElement('strong');
      title.textContent = trigger.dataset.title || '';
      lightboxCap.appendChild(title);
      lightboxCap.appendChild(document.createTextNode(trigger.dataset.caption || ''));

      lightbox.hidden = false;
      document.body.style.overflow = 'hidden';
      requestAnimationFrame(function () { lightbox.classList.add('is-open'); });
      lightboxClose.focus();
    };

    document.querySelectorAll('.fig-media').forEach(function (btn) {
      /* Without this the button's only accessible name is the raw figure alt,
         which never says that activating it opens anything. */
      btn.setAttribute('aria-label', 'Open figure full size: ' + (btn.dataset.title || ''));
      btn.addEventListener('click', function () { openLightbox(btn); });
    });

    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox || e.target === lightboxImg.parentNode) closeLightbox();
    });
  }

  document.addEventListener('keydown', function (e) {
    var lbOpen = lightbox && !lightbox.hidden;
    if (e.key === 'Escape') {
      if (lbOpen) closeLightbox();
      else if (navMenu && navMenu.classList.contains('is-open')) { closeMenu(); navToggle.focus(); }
    }
    /* Keep focus on the close button while the dialog is open. */
    if (e.key === 'Tab' && lbOpen) {
      e.preventDefault();
      lightboxClose.focus();
    }
  });

  /* ---------- CV link, only if the file is really there ---------- */
  var cvLink = document.getElementById('cv-link');
  if (cvLink && window.fetch) {
    fetch(cvLink.getAttribute('href'), { method: 'HEAD' })
      .then(function (res) {
        var type = res.headers.get('content-type') || '';
        if (res.ok && type.indexOf('html') === -1) cvLink.hidden = false;
      })
      .catch(function () { /* leave hidden */ });
  }

  /* ---------- Footer year ---------- */
  var year = document.getElementById('year');
  if (year) year.textContent = new Date().getFullYear();
})();
