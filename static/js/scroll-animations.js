/**
 * scroll-animations.js — NoteGuard Scroll-Triggered Fade & Slide Animations
 * Vanilla JS with IntersectionObserver, scroll direction detection (bidirectional),
 * staggered child animation support, dynamic mutation observation, and prefers-reduced-motion support.
 */

(function () {
  'use strict';

  // Check if reduced motion is preferred
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Track previous scroll position to determine scroll direction
  let lastScrollY = window.pageYOffset || document.documentElement.scrollTop;
  let scrollDirection = 'down'; // 'down' or 'up'

  window.addEventListener('scroll', () => {
    const currentScrollY = window.pageYOffset || document.documentElement.scrollTop;
    if (currentScrollY > lastScrollY) {
      scrollDirection = 'down';
    } else if (currentScrollY < lastScrollY) {
      scrollDirection = 'up';
    }
    lastScrollY = Math.max(0, currentScrollY);
  }, { passive: true });

  // IntersectionObserver configuration
  const observerOptions = {
    threshold: 0.15,
    rootMargin: '0px 0px -8% 0px'
  };

  let scrollObserver = null;

  function handleIntersection(entries) {
    if (prefersReducedMotion) {
      entries.forEach(entry => {
        entry.target.classList.add('is-visible');
        entry.target.classList.remove('scroll-enter-up');
      });
      return;
    }

    entries.forEach(entry => {
      const el = entry.target;

      if (entry.isIntersecting) {
        // Tag direction before making visible so the animation starts from top or bottom
        if (scrollDirection === 'up') {
          el.classList.add('scroll-enter-up');
        } else {
          el.classList.remove('scroll-enter-up');
        }

        // Apply delay if configured via data-delay attribute or dynamic stagger
        const delay = el.getAttribute('data-delay');
        if (delay) {
          el.style.transitionDelay = `${delay}ms`;
        }

        // Add visibility class
        el.classList.add('is-visible');
      } else {
        // When leaving viewport, remove visibility class so it can animate again when scrolled back
        el.classList.remove('is-visible');
        // Clean up temporary inline transition delay after exit
        if (el.getAttribute('data-delay')) {
          el.style.transitionDelay = '';
        }
      }
    });
  }

  function setupStaggerGroups() {
    // Find all containers with data-animate-group
    const groups = document.querySelectorAll('[data-animate-group]');
    groups.forEach(group => {
      // Find direct or immediate animated children
      const children = group.querySelectorAll(':scope > [data-animate], :scope > * > [data-animate]');
      children.forEach((child, index) => {
        if (!child.hasAttribute('data-delay')) {
          // Stagger by ~80ms, capped at 600ms
          const stagger = Math.min(index * 80, 600);
          child.setAttribute('data-delay', stagger);
        }
      });
    });
  }

  function observeElements() {
    if (!scrollObserver) return;

    setupStaggerGroups();

    const animElements = document.querySelectorAll('[data-animate]');
    animElements.forEach(el => {
      scrollObserver.observe(el);
    });
  }

  function init() {
    if (!('IntersectionObserver' in window)) {
      // Fallback for browsers without IntersectionObserver
      document.querySelectorAll('[data-animate]').forEach(el => {
        el.classList.add('is-visible');
      });
      return;
    }

    scrollObserver = new IntersectionObserver(handleIntersection, observerOptions);
    observeElements();

    // Observe dynamically injected nodes (e.g. analysis results, warnings, checklist rows)
    const mutationObserver = new MutationObserver((mutations) => {
      let hasNewElements = false;
      mutations.forEach(mutation => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
          mutation.addedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              if (node.hasAttribute('data-animate') || node.querySelector('[data-animate]')) {
                hasNewElements = true;
              }
            }
          });
        }
      });

      if (hasNewElements) {
        window.refreshScrollAnimations();
      }
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  // Exported global method to manually refresh and register dynamically created elements
  window.refreshScrollAnimations = function () {
    observeElements();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
