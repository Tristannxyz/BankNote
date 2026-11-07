/**
 * how-to-use.js — NoteGuard "How to Use" Interactive Guide Modal Controller
 * Accessible modal dialog with stepper, dot navigation, keyboard accessibility (Arrow keys, Esc, Tab trapping),
 * scroll lock, responsive behavior, first-visit pulsing dot indicator, and prefers-reduced-motion support.
 */

(function () {
  'use strict';

  // Constants & Storage Keys
  const STORAGE_KEY_SEEN = 'noteguard_how_to_use_seen';

  // DOM Elements
  let modalBackdrop = null;
  let modalDialog = null;
  let btnOpen = null;
  let btnClose = null;
  let btnPrev = null;
  let btnNext = null;
  let btnDone = null;
  let btnGoVerify = null;
  let stepperDotsContainer = null;
  let stepCounter = null;
  let stepSlides = [];

  let currentStep = 0;
  let totalSteps = 6;
  let lastFocusedElement = null;

  function hasSeenGuide() {
    try {
      return localStorage.getItem(STORAGE_KEY_SEEN) === 'true';
    } catch (e) {
      return false;
    }
  }

  function markGuideAsSeen() {
    try {
      localStorage.setItem(STORAGE_KEY_SEEN, 'true');
    } catch (e) {}

    if (btnOpen) {
      btnOpen.classList.remove('has-pulse-dot');
    }
  }

  function updateStep(stepIndex) {
    if (stepIndex < 0) stepIndex = 0;
    if (stepIndex >= totalSteps) stepIndex = totalSteps - 1;
    currentStep = stepIndex;

    // Update slides
    stepSlides.forEach((slide, idx) => {
      if (idx === currentStep) {
        slide.classList.add('active');
        slide.removeAttribute('hidden');
      } else {
        slide.classList.remove('active');
        slide.setAttribute('hidden', 'true');
      }
    });

    // Update dots
    if (stepperDotsContainer) {
      const dots = stepperDotsContainer.querySelectorAll('.guide-step-dot');
      dots.forEach((dot, idx) => {
        const isActive = idx === currentStep;
        dot.classList.toggle('active', isActive);
        dot.setAttribute('aria-current', isActive ? 'step' : 'false');
      });
    }

    // Update Counter Text
    if (stepCounter) {
      stepCounter.textContent = `Step ${currentStep + 1} of ${totalSteps}`;
    }

    // Update buttons visibility
    if (btnPrev) {
      btnPrev.disabled = currentStep === 0;
      btnPrev.style.visibility = currentStep === 0 ? 'hidden' : 'visible';
    }

    const isLastStep = currentStep === totalSteps - 1;
    if (btnNext) {
      btnNext.style.display = isLastStep ? 'none' : 'inline-flex';
    }
    if (btnDone) {
      btnDone.style.display = isLastStep ? 'inline-flex' : 'none';
    }
  }

  function openModal() {
    if (!modalBackdrop) return;

    lastFocusedElement = document.activeElement;
    markGuideAsSeen();

    modalBackdrop.classList.add('is-open');
    modalBackdrop.removeAttribute('hidden');
    document.body.classList.add('guide-modal-open');

    updateStep(0);

    // Shift focus into modal (first focusable element or close button)
    setTimeout(() => {
      if (btnClose) {
        btnClose.focus();
      } else if (modalDialog) {
        modalDialog.focus();
      }
    }, 50);

    document.addEventListener('keydown', handleKeyDown);
  }

  function closeModal() {
    if (!modalBackdrop) return;

    modalBackdrop.classList.remove('is-open');
    modalBackdrop.setAttribute('hidden', 'true');
    document.body.classList.remove('guide-modal-open');

    document.removeEventListener('keydown', handleKeyDown);

    // Return focus
    if (lastFocusedElement && typeof lastFocusedElement.focus === 'function') {
      lastFocusedElement.focus();
    } else if (btnOpen) {
      btnOpen.focus();
    }
  }

  function handleKeyDown(e) {
    if (!modalBackdrop || !modalBackdrop.classList.contains('is-open')) return;

    // Close on Escape
    if (e.key === 'Escape') {
      e.preventDefault();
      closeModal();
      return;
    }

    // Arrow keys navigation between steps
    if (e.key === 'ArrowRight') {
      if (currentStep < totalSteps - 1) {
        e.preventDefault();
        updateStep(currentStep + 1);
      }
      return;
    }
    if (e.key === 'ArrowLeft') {
      if (currentStep > 0) {
        e.preventDefault();
        updateStep(currentStep - 1);
      }
      return;
    }

    // Tab key focus trap inside modal
    if (e.key === 'Tab') {
      const focusableElements = modalDialog.querySelectorAll(
        'button:not([disabled]):not([hidden]):not([style*="display: none"]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      if (focusableElements.length === 0) return;

      const firstEl = focusableElements[0];
      const lastEl = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstEl) {
          e.preventDefault();
          lastEl.focus();
        }
      } else {
        if (document.activeElement === lastEl) {
          e.preventDefault();
          firstEl.focus();
        }
      }
    }
  }

  function navigateToVerify() {
    closeModal();
    const isIndex = window.location.pathname === '/' ||
                    window.location.pathname.endsWith('/index') ||
                    window.location.pathname.endsWith('/index.html') ||
                    window.location.pathname === '';

    if (isIndex) {
      const uploadCard = document.querySelector('.upload-main-card') || document.getElementById('dropZone');
      if (uploadCard) {
        uploadCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    } else {
      window.location.href = '/';
    }
  }

  function init() {
    modalBackdrop = document.getElementById('howToUseModal');
    if (!modalBackdrop) return;

    modalDialog = modalBackdrop.querySelector('.guide-modal-dialog');
    btnOpen = document.getElementById('btnHowToUse');
    btnClose = document.getElementById('btnCloseGuideModal');
    btnPrev = document.getElementById('btnGuidePrev');
    btnNext = document.getElementById('btnGuideNext');
    btnDone = document.getElementById('btnGuideDone');
    btnGoVerify = document.getElementById('btnGuideGoVerify');
    stepperDotsContainer = document.getElementById('guideStepperDots');
    stepCounter = document.getElementById('guideStepCounter');

    stepSlides = Array.from(modalBackdrop.querySelectorAll('.guide-step-slide'));
    totalSteps = stepSlides.length || 6;

    // Show pulsing dot on first visit only
    if (!hasSeenGuide() && btnOpen) {
      btnOpen.classList.add('has-pulse-dot');
    }

    // Bind event listeners
    if (btnOpen) {
      btnOpen.addEventListener('click', openModal);
    }
    if (btnClose) {
      btnClose.addEventListener('click', closeModal);
    }
    if (btnPrev) {
      btnPrev.addEventListener('click', () => updateStep(currentStep - 1));
    }
    if (btnNext) {
      btnNext.addEventListener('click', () => updateStep(currentStep + 1));
    }
    if (btnDone) {
      btnDone.addEventListener('click', closeModal);
    }
    if (btnGoVerify) {
      btnGoVerify.addEventListener('click', navigateToVerify);
    }

    // Backdrop click close
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) {
        closeModal();
      }
    });

    // Step dots click jump
    if (stepperDotsContainer) {
      const dots = stepperDotsContainer.querySelectorAll('.guide-step-dot');
      dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
          updateStep(index);
        });
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

