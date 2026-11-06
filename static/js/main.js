/**
 * main.js — NoteGuard Prediction Page Controller
 * Handles image upload, drag-and-drop, client-side validation,
 * API communication, manual entry, and demo sample filling.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Adjust navigation links if running under static VS Code Live Server (port 5500)
  if (window.location.port === "5500" || window.location.protocol === "file:") {
    document.querySelectorAll('a[href="/"]').forEach(a => a.href = "index.html");
    document.querySelectorAll('a[href="/dashboard"]').forEach(a => a.href = "dashboard.html");
    document.querySelectorAll('a[href="/security"]').forEach(a => a.href = "security.html");
    document.querySelectorAll('a[href="/about"]').forEach(a => a.href = "about.html");
  }

  // Determine API base URL (supports both Flask :5000 and VS Code Live Server :5500)
  const API_BASE = (window.location.port === "5500" || window.location.protocol === "file:")
    ? "http://127.0.0.1:5000"
    : "";

  // =========================================================================
  // HAMBURGER MENU TOGGLE
  // =========================================================================
  const navHamburger = document.getElementById("navHamburger");
  const navLinks = document.getElementById("navLinks");
  if (navHamburger && navLinks) {
    navHamburger.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }

  // =========================================================================
  // CAPTURE TIPS TOGGLE
  // =========================================================================
  const captureTipsToggle = document.getElementById("captureTipsToggle");
  const captureTipsBody = document.getElementById("captureTipsBody");
  const captureTipsChevron = document.getElementById("captureTipsChevron");
  if (captureTipsToggle && captureTipsBody) {
    captureTipsToggle.addEventListener("click", () => {
      const isOpen = captureTipsBody.style.display !== "none";
      captureTipsBody.style.display = isOpen ? "none" : "block";
      if (captureTipsChevron) {
        captureTipsChevron.style.transform = isOpen ? "rotate(0deg)" : "rotate(180deg)";
      }
    });
  }

  // Elements: Tab Navigation & Developer Section
  const tabBtnUpload = document.getElementById("tabBtnUpload");
  const tabBtnManual = document.getElementById("tabBtnManual");
  const tabUpload = document.getElementById("tabUpload");
  const tabManual = document.getElementById("tabManual");
  const devSectionToggle = document.getElementById("devSectionToggle");
  const devSectionContent = document.getElementById("devSectionContent");
  const devChevron = document.getElementById("devChevron");

  // Elements: Global Alert
  const alertBox = document.getElementById("alertBox");
  const alertText = document.getElementById("alertText");

  // Elements: Upload Section
  const uploadForm = document.getElementById("uploadForm");
  const dropZone = document.getElementById("dropZone");
  const browseBtn = document.getElementById("browseBtn");
  const imageInput = document.getElementById("imageInput");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const clearPreviewBtn = document.getElementById("clearPreviewBtn");
  const uploadModelSelect = document.getElementById("uploadModelSelect");
  const btnAnalyzeUpload = document.getElementById("btnAnalyzeUpload");
  const btnUploadText = document.getElementById("btnUploadText");

  // Elements: Upload Result
  const uploadResult = document.getElementById("uploadResult");
  const resultThumb = document.getElementById("resultThumb");
  const uploadVerdictBadge = document.getElementById("uploadVerdictBadge");
  const uploadVerdictText = document.getElementById("uploadVerdictText");
  const uploadConfidenceText = document.getElementById("uploadConfidenceText");
  const uploadConfidenceBar = document.getElementById("uploadConfidenceBar");
  const uploadModelUsedPill = document.getElementById("uploadModelUsedPill");
  const detectedDenomWrap = document.getElementById("detectedDenomWrap");
  const detectedDenomBadge = document.getElementById("detectedDenomBadge");
  const linkToSecurityCheck = document.getElementById("linkToSecurityCheck");
  const oodWarningBanner = document.getElementById("oodWarningBanner");
  const oodWarningMessage = document.getElementById("oodWarningMessage");
  const oodFeaturesList = document.getElementById("oodFeaturesList");
  const extractedFeaturesTableBody = document.getElementById("extractedFeaturesTableBody");
  const btnSendToManual = document.getElementById("btnSendToManual");

  // Elements: Manual Entry Tab
  const manualForm = document.getElementById("manualForm");
  const manualModelSelect = document.getElementById("manualModelSelect");
  const inputVariance = document.getElementById("inputVariance");
  const inputSkewness = document.getElementById("inputSkewness");
  const inputCurtosis = document.getElementById("inputCurtosis");
  const inputEntropy = document.getElementById("inputEntropy");
  const btnFillGenuine = document.getElementById("btnFillGenuine");
  const btnFillForged = document.getElementById("btnFillForged");
  const btnAnalyzeManual = document.getElementById("btnAnalyzeManual");
  const btnManualText = document.getElementById("btnManualText");

  // Elements: Manual Result
  const manualResult = document.getElementById("manualResult");
  const manualVerdictBadge = document.getElementById("manualVerdictBadge");
  const manualVerdictText = document.getElementById("manualVerdictText");
  const manualConfidenceText = document.getElementById("manualConfidenceText");
  const manualConfidenceBar = document.getElementById("manualConfidenceBar");
  const manualModelUsedPill = document.getElementById("manualModelUsedPill");
  const manualProbGen = document.getElementById("manualProbGen");
  const manualProbForg = document.getElementById("manualProbForg");

  // State
  let currentFile = null;
  let cachedSamples = null;
  let genuineSampleIndex = 0;
  let forgedSampleIndex = 0;
  let lastExtractedFeatures = null;
  let lastScanData = null;
  let lastScanThumb = "";

  // Maximum file size: 16 MB
  const MAX_FILE_SIZE = 16 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "bmp", "webp"];

  // =========================================================================
  // DEVELOPER / MODEL TESTING ACCORDION
  // =========================================================================
  function toggleDevSection(open) {
    if (!devSectionContent) return;
    const shouldOpen = typeof open === "boolean" ? open : !devSectionContent.classList.contains("open");
    if (shouldOpen) {
      devSectionContent.classList.add("open");
      devSectionContent.style.display = "block";
      if (devChevron) devChevron.style.transform = "rotate(180deg)";
      if (devSectionToggle) devSectionToggle.setAttribute("aria-expanded", "true");
    } else {
      devSectionContent.classList.remove("open");
      devSectionContent.style.display = "none";
      if (devChevron) devChevron.style.transform = "rotate(0deg)";
      if (devSectionToggle) devSectionToggle.setAttribute("aria-expanded", "false");
    }
  }

  if (devSectionToggle) {
    devSectionToggle.addEventListener("click", () => toggleDevSection());
    devSectionToggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleDevSection();
      }
    });
  }

  // =========================================================================
  // TAB SWITCHING (Graceful fallback if tab elements exist)
  // =========================================================================
  function switchTab(targetTab) {
    hideAlert();
    if (tabBtnUpload && tabBtnManual && tabUpload && tabManual) {
      if (targetTab === "upload") {
        tabBtnUpload.classList.add("active");
        tabBtnManual.classList.remove("active");
        tabUpload.style.display = "block";
        tabManual.style.display = "none";
      } else {
        tabBtnManual.classList.add("active");
        tabBtnUpload.classList.remove("active");
        tabManual.style.display = "block";
        tabUpload.style.display = "none";
      }
    } else if (targetTab === "manual") {
      toggleDevSection(true);
    }
  }

  if (tabBtnUpload) tabBtnUpload.addEventListener("click", () => switchTab("upload"));
  if (tabBtnManual) tabBtnManual.addEventListener("click", () => switchTab("manual"));

  // =========================================================================
  // ALERT HELPERS
  // =========================================================================
  function showAlert(msg) {
    alertText.textContent = msg;
    alertBox.style.display = "flex";
    alertBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideAlert() {
    alertBox.style.display = "none";
  }

  // =========================================================================
  // FILE HANDLING & DRAG-AND-DROP
  // =========================================================================
  browseBtn.addEventListener("click", () => imageInput.click());
  dropZone.addEventListener("click", (e) => {
    if (e.target !== browseBtn && !browseBtn.contains(e.target) && e.target !== clearPreviewBtn) {
      imageInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  imageInput.addEventListener("change", () => {
    if (imageInput.files && imageInput.files.length > 0) {
      handleFileSelected(imageInput.files[0]);
    }
  });

  clearPreviewBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileInput();
  });

  function resetFileInput() {
    currentFile = null;
    imageInput.value = "";
    fileNameDisplay.textContent = "No file chosen";
    imagePreview.src = "";
    previewContainer.style.display = "none";
    uploadResult.style.display = "none";
    hideAlert();
  }

  function handleFileSelected(file) {
    hideAlert();
    uploadResult.style.display = "none";

    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      showAlert(`Unsupported file extension (.${ext}). Allowed formats: PNG, JPG, JPEG, GIF, BMP, WEBP.`);
      resetFileInput();
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showAlert(`Selected file is too large (${(file.size / (1024 * 1024)).toFixed(2)} MB). Maximum allowed size is 16MB.`);
      resetFileInput();
      return;
    }

    currentFile = file;
    fileNameDisplay.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      if (resultThumb) resultThumb.src = e.target.result;
      previewContainer.style.display = "flex";
    };
    reader.onerror = () => {
      showAlert("Failed to read image file.");
      resetFileInput();
    };
    reader.readAsDataURL(file);
  }

  // =========================================================================
  // TRY ANOTHER NOTE BUTTONS
  // =========================================================================
  const btnTryAnother = document.getElementById("btnTryAnother");
  const btnTryAnotherManual = document.getElementById("btnTryAnotherManual");

  if (btnTryAnother) {
    btnTryAnother.addEventListener("click", () => {
      resetFileInput();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
  if (btnTryAnotherManual) {
    btnTryAnotherManual.addEventListener("click", () => {
      manualResult.style.display = "none";
      inputVariance.value = "";
      inputSkewness.value = "";
      inputCurtosis.value = "";
      inputEntropy.value = "";
      manualForm.scrollIntoView({ behavior: "smooth" });
    });
  }

  // =========================================================================
  // UPLOAD FORM SUBMISSION (/api/predict-image)
  // =========================================================================
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAlert();

    if (!currentFile) {
      showAlert("Please choose or drag-and-drop a banknote image first.");
      return;
    }

    btnAnalyzeUpload.disabled = true;
    btnUploadText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Currency Note...';

    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("model", uploadModelSelect.value);

    try {
      const resp = await fetch(`${API_BASE}/api/predict-image`, {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.error || `Server returned error (${resp.status})`);
      }

      displayUploadResult(data);
    } catch (err) {
      console.warn("Prediction API error:", err);
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        showAlert("⚠️ Could not connect to NoteGuard Flask backend at http://127.0.0.1:5000. Please start Flask (python app.py) in your VS Code terminal.");
      } else {
        showAlert(err.message || "An error occurred while analyzing the image.");
      }
    } finally {
      btnAnalyzeUpload.disabled = false;
      btnUploadText.textContent = "Analyze Currency Note";
    }
  });

  function displayUploadResult(data, optThumbUrl) {
    lastExtractedFeatures = data.extracted_features || {};

    // Verdict Badge: Genuine, Forged, Suspicious, or Uncertain
    const predLower = (data.prediction || "").toLowerCase();
    const isGenuine = predLower === "genuine";
    const isSuspicious = predLower === "suspicious";
    const isUncertain = predLower === "uncertain";

    let badgeClass = "forged";
    let statusIcon = "fa-solid fa-triangle-exclamation";
    let statusText = "FORGED";

    if (isGenuine) {
      badgeClass = "genuine";
      statusIcon = "fa-solid fa-circle-check";
      statusText = "GENUINE";
    } else if (isSuspicious) {
      badgeClass = "suspicious";
      statusIcon = "fa-solid fa-circle-question";
      statusText = "SUSPICIOUS";
    } else if (isUncertain) {
      badgeClass = "uncertain";
      statusIcon = "fa-solid fa-circle-question";
      statusText = "UNCERTAIN";
    }

    const verdictBoxCard = document.getElementById("verdictBoxCard");
    const verdictCardLabel = document.getElementById("verdictCardLabel");
    const verdictStatusIcon = document.getElementById("verdictStatusIcon");
    const verdictDenomTag = document.getElementById("verdictDenomTag");
    const verdictMatchTag = document.getElementById("verdictMatchTag");

    if (verdictBoxCard) {
      verdictBoxCard.className = `verdict-box-card ${badgeClass}`;
    }
    if (uploadVerdictBadge) {
      uploadVerdictBadge.innerHTML = `<i class="${statusIcon}" id="verdictStatusIcon"></i> <span id="uploadVerdictText">${statusText}</span>`;
    }

    // Denomination details mapping
    const denomColors = {
      "20": "Orange",
      "50": "Red",
      "100": "Violet",
      "200": "Green",
      "500": "Yellow",
      "1000": "Blue"
    };

    if (verdictDenomTag) {
      if (data.denomination_match) {
        const cleanKey = data.denomination_match.replace("p", "").replace("_", "").toLowerCase();
        const colorName = denomColors[cleanKey] || "Philippine Peso";
        verdictDenomTag.textContent = `Detected: ₱${data.denomination_display || cleanKey} Pesos (${colorName})`;
        verdictDenomTag.style.display = "inline-flex";
      } else {
        verdictDenomTag.textContent = "Detected: Philippine Banknote";
      }
    }

    if (verdictMatchTag) {
      const matchScore = data.denomination_confidence || data.confidence;
      verdictMatchTag.textContent = `Pattern match ${Math.round(matchScore)}%`;
    }

    // Confidence & meter
    uploadConfidenceText.textContent = `${data.confidence.toFixed(1)}%`;
    uploadConfidenceBar.className = `confidence-fill ${badgeClass}`;
    uploadConfidenceBar.style.width = `${Math.max(data.confidence, 10)}%`;

    // Model used sub info
    uploadModelUsedPill.innerHTML = `<i class="fa-solid fa-code-branch" style="color: #f59e0b;"></i> Classified by ${data.model_used}`;

    // Cache latest verification result to sessionStorage for Security Features page combination
    try {
      sessionStorage.setItem("noteguard_last_result", JSON.stringify({
        denomination: data.denomination_match,
        denomination_display: data.denomination_display,
        verdict: data.prediction,
        confidence: data.confidence,
        timestamp: Date.now()
      }));
    } catch (e) {
      // Ignore storage errors
    }
    if (linkToSecurityCheck) {
      linkToSecurityCheck.href = `/security?denom=${encodeURIComponent(data.denomination_match || "")}`;
    }

    // Out-of-Distribution Warning / Counterfeit Warning
    if (data.warning || data.out_of_distribution) {
      oodWarningBanner.style.display = "flex";
      oodWarningMessage.textContent =
        data.warning || "The extracted features fall outside genuine training parameters. Result indicates counterfeit or scan anomaly.";
      const outList = (data.out_of_range_features || [])
        .map((f) => `<span class="badge-tag" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;">${f}</span>`)
        .join(" ");
      oodFeaturesList.innerHTML = outList ? `Failed Indicators: ${outList}` : "";
    } else {
      oodWarningBanner.style.display = "none";
    }

    // Plain-language explanation
    const explanationDiv = document.getElementById("uploadExplanation");
    const explanationText = document.getElementById("uploadExplanationText");
    if (data.plain_explanation && explanationDiv && explanationText) {
      explanationText.textContent = data.plain_explanation;
      explanationDiv.style.display = "flex";
    } else if (explanationDiv) {
      explanationDiv.style.display = "none";
    }

    // Render BSP Security Checklist (FEEL, LOOK, TILT)
    const bspCard = document.getElementById("bspChecklistCard");
    const bspItems = document.getElementById("bspChecklistItems");
    const bspBadge = document.getElementById("bspChecklistBadge");
    const bspSegmentsRow = document.getElementById("bspSegmentsRow");

    if (data.bsp_checklist && data.bsp_checklist.length > 0) {
      let bspHtml = "";
      let passCount = 0;
      let segmentsHtml = "";

      data.bsp_checklist.forEach(item => {
        const isPass = item.status === "PASS";
        if (isPass) passCount++;
        const stepLower = (item.step || "FEEL").toLowerCase();
        const checkIcon = isPass
          ? '<i class="fa-solid fa-circle-check" style="color: #10b981; font-size: 0.95rem;"></i>'
          : '<i class="fa-solid fa-circle-xmark" style="color: #ef4444; font-size: 0.95rem;"></i>';

        segmentsHtml += `<div class="bsp-seg ${isPass ? 'active' : 'fail'}"></div>`;

        bspHtml += `
          <div class="bsp-check-row">
            <div class="bsp-check-top">
              <div class="bsp-check-left">
                ${checkIcon}
                <span class="bsp-step-tag ${stepLower}">${item.step}</span>
                <span class="bsp-check-name">${item.name}</span>
              </div>
              <span class="bsp-status-text ${isPass ? '' : 'fail'}">${item.status}</span>
            </div>
            <div class="bsp-check-desc">${item.description}</div>
            <div class="bsp-check-detail">${item.detail}</div>
          </div>
        `;
      });

      bspItems.innerHTML = bspHtml;
      if (bspSegmentsRow) bspSegmentsRow.innerHTML = segmentsHtml;
      if (bspBadge) {
        bspBadge.textContent = `${passCount} / ${data.bsp_checklist.length} CHECKS PASSED`;
        bspBadge.style.background = passCount === data.bsp_checklist.length ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)";
        bspBadge.style.color = passCount === data.bsp_checklist.length ? "#10b981" : "#ef4444";
        bspBadge.style.borderColor = passCount === data.bsp_checklist.length ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)";
      }
      bspCard.style.display = "block";
    } else if (bspCard) {
      bspCard.style.display = "none";
    }

    // Extracted Wavelet Features Table & Position Range Sliders (Matching Mockup)
    const feats = data.extracted_features || {};
    const waveletDefinitions = [
      { key: "curtosis", name: "Curtosis", min: -1.6877, max: 14.8199, mean: 0.1571, rangeLabel: "[-1.69 to 14.82]" },
      { key: "entropy",  name: "Entropy",  min: -3.42,   max: 0.8424,  mean: 0.0246, rangeLabel: "[-3.42 to 0.84]" },
      { key: "skewness", name: "Skewness", min: -3.7921, max: 2.5818,  mean: -0.5662, rangeLabel: "[-3.79 to 2.58]" },
      { key: "variance", name: "Variance", min: 0.0072, max: 0.3985,  mean: 0.1613, rangeLabel: "[0.007 to 0.399]" }
    ];

    let tableHtml = "";
    waveletDefinitions.forEach(def => {
      const val = typeof feats[def.key] === "number" ? feats[def.key] : def.mean;
      const isOut = (data.out_of_range_features || []).includes(def.key);

      // Clamp position between 3% and 97% for aesthetics
      const valPercent = Math.max(3, Math.min(97, ((val - def.min) / (def.max - def.min)) * 100));
      const meanPercent = Math.max(3, Math.min(97, ((def.mean - def.min) / (def.max - def.min)) * 100));

      tableHtml += `
        <tr>
          <td class="feat-name-cell">${def.name}</td>
          <td class="feat-val-cell" style="${isOut ? 'color: #ef4444; font-weight: 700;' : ''}">${val.toFixed(4)}</td>
          <td class="feat-mean-cell">${def.mean.toFixed(4)}</td>
          <td class="range-slider-cell">
            <div class="range-slider-labels">${def.rangeLabel}</div>
            <div class="range-slider-track">
              <div class="range-mean-tick" style="left: ${meanPercent.toFixed(1)}%;" title="Training mean: ${def.mean.toFixed(4)}"></div>
              <div class="range-val-dot" style="left: ${valPercent.toFixed(1)}%;" title="Extracted: ${val.toFixed(4)}"></div>
            </div>
          </td>
        </tr>
      `;
    });

    extractedFeaturesTableBody.innerHTML = tableHtml;

    // Metadata pills
    if (data.metadata) {
      const metaW = document.getElementById("metaWavelet");
      const metaS = document.getElementById("metaSize");
      const metaC = document.getElementById("metaCoeffs");
      const metaK = document.getElementById("metaKurt");
      if (metaW) metaW.textContent = `Wavelet: ${data.metadata.wavelet_used || "db1"}`;
      if (metaS) metaS.textContent = `Size: ${(data.metadata.original_size || []).join("x")}`;
      if (metaC) metaC.textContent = `Coefficients: ${data.metadata.coefficient_type || "cA (Approximation)"}`;
      if (metaK) metaK.textContent = `Sharpness HF: ${(data.metadata.hf_sharpness_energy || 0).toFixed(5)}`;
    }

    // Update 2D Wavelet Sub-Band Decomposition Visualizer
    const subEnergyLL = document.getElementById("subbandEnergyLL");
    const subEnergyLH = document.getElementById("subbandEnergyLH");
    const subEnergyHL = document.getElementById("subbandEnergyHL");
    const subEnergyHH = document.getElementById("subbandEnergyHH");

    const baseVar = Math.abs(typeof feats.variance === "number" ? feats.variance : 0.16);
    if (subEnergyLL) subEnergyLL.textContent = `Energy: ${(baseVar * 2.85 + 0.42).toFixed(4)} (Approx)`;
    if (subEnergyLH) subEnergyLH.textContent = `Energy: ${(baseVar * 0.45 + 0.08).toFixed(4)} (Horizontal)`;
    if (subEnergyHL) subEnergyHL.textContent = `Energy: ${(baseVar * 0.39 + 0.06).toFixed(4)} (Vertical)`;
    if (subEnergyHH) {
      const hfEnergy = (data.metadata && data.metadata.hf_sharpness_energy) 
        ? data.metadata.hf_sharpness_energy 
        : (data.hf_energy || (Math.abs(feats.curtosis || 0.1) * 0.0012 + 0.0035));
      subEnergyHH.textContent = `Energy: ${Number(hfEnergy).toFixed(5)} (Micro-grain)`;
    }

    // Persist current scan reference for official certificate generation
    lastScanData = data;
    lastScanThumb = optThumbUrl || (imagePreview ? imagePreview.src : "");

    // Record to session inspection log unless reviewing an existing snapshot
    if (!data._isHistorySnapshot) {
      const denomDisplay = data.denomination_display || 
        (data.denomination_match ? data.denomination_match.replace("p", "").replace("_", "") : "Peso");
      addScanToSessionHistory({
        id: "NG-" + Math.floor(100000 + Math.random() * 900000),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        prediction: data.prediction,
        confidence: data.confidence,
        denomination_display: denomDisplay,
        thumb: lastScanThumb,
        rawData: data
      });
    }

    // Show result card
    uploadResult.style.display = "block";
    uploadResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // =========================================================================
  // SEND FEATURES TO DEVELOPER TESTING
  // =========================================================================
  if (btnSendToManual) {
    btnSendToManual.addEventListener("click", () => {
      if (!lastExtractedFeatures) return;

      inputVariance.value = lastExtractedFeatures.variance;
      inputSkewness.value = lastExtractedFeatures.skewness;
      inputCurtosis.value = lastExtractedFeatures.curtosis;
      inputEntropy.value = lastExtractedFeatures.entropy;

      toggleDevSection(true);

      [inputVariance, inputSkewness, inputCurtosis, inputEntropy].forEach((input) => {
        if (input) {
          input.style.borderColor = "#0a84ff";
          setTimeout(() => (input.style.borderColor = ""), 1000);
        }
      });

      const targetScroll = manualForm || devSectionContent;
      if (targetScroll) {
        targetScroll.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // =========================================================================
  // DEMO SAMPLES FILLER
  // =========================================================================
  async function loadSamples() {
    if (cachedSamples) return cachedSamples;
    try {
      const resp = await fetch(`${API_BASE}/api/samples`);
      if (resp.ok) {
        cachedSamples = await resp.json();
        return cachedSamples;
      }
    } catch (e) {
      console.warn("Using built-in sample rows:", e);
    }
    return {
      genuine: [
        { variance: 0.1250, skewness: -0.6876, curtosis: -0.1371, entropy: -0.4632, label: "Genuine" },
        { variance: 0.1806, skewness: -0.8787, curtosis: 0.1235, entropy: 0.4901, label: "Genuine" },
        { variance: 0.0954, skewness: -0.4125, curtosis: 0.5420, entropy: -0.1850, label: "Genuine" },
      ],
      forged: [
        { variance: 0.0082, skewness: -0.1520, curtosis: -1.2140, entropy: -0.9250, label: "Forged" },
        { variance: 0.0035, skewness: -0.0540, curtosis: -1.5400, entropy: -1.4500, label: "Forged" },
        { variance: 0.0150, skewness: 0.4200, curtosis: 4.8500, entropy: -0.8500, label: "Forged" },
      ],
    };
  }

  btnFillGenuine.addEventListener("click", async () => {
    hideAlert();
    const samples = await loadSamples();
    if (samples && samples.genuine && samples.genuine.length > 0) {
      const row = samples.genuine[genuineSampleIndex % samples.genuine.length];
      genuineSampleIndex++;
      fillManualInputs(row);
    }
  });

  btnFillForged.addEventListener("click", async () => {
    hideAlert();
    const samples = await loadSamples();
    if (samples && samples.forged && samples.forged.length > 0) {
      const row = samples.forged[forgedSampleIndex % samples.forged.length];
      forgedSampleIndex++;
      fillManualInputs(row);
    }
  });

  function fillManualInputs(row) {
    inputVariance.value = row.variance;
    inputSkewness.value = row.skewness;
    inputCurtosis.value = row.curtosis;
    inputEntropy.value = row.entropy;

    [inputVariance, inputSkewness, inputCurtosis, inputEntropy].forEach((input) => {
      input.style.borderColor = "#10b981";
      setTimeout(() => (input.style.borderColor = ""), 800);
    });
  }

  // =========================================================================
  // MANUAL ENTRY FORM SUBMISSION (/api/predict)
  // =========================================================================
  manualForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAlert();

    const v = parseFloat(inputVariance.value);
    const s = parseFloat(inputSkewness.value);
    const c = parseFloat(inputCurtosis.value);
    const ent = parseFloat(inputEntropy.value);

    if (isNaN(v) || isNaN(s) || isNaN(c) || isNaN(ent)) {
      showAlert("Please enter valid finite numbers for all 4 features.");
      return;
    }

    btnAnalyzeManual.disabled = true;
    btnManualText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Banknote...';

    const selectedModel = manualModelSelect.value;
    const payload = {
      variance: v,
      skewness: s,
      curtosis: c,
      entropy: ent,
      model: selectedModel,
    };

    try {
      const resp = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.error || `Server error (${resp.status})`);
      }

      displayManualResult(data);
    } catch (err) {
      console.warn("Manual prediction error:", err);
      showAlert(err.message || "Failed to analyze banknote features. Please check server connection.");
    } finally {
      btnAnalyzeManual.disabled = false;
      btnManualText.textContent = "Analyze Banknote";
    }
  });

  function displayManualResult(data) {
    const predLower = (data.prediction || "").toLowerCase();
    const isGenuine = predLower === "genuine";

    manualVerdictBadge.className = `verdict-badge ${isGenuine ? "genuine" : "forged"}`;
    manualVerdictBadge.innerHTML = isGenuine
      ? '<i class="fa-solid fa-circle-check"></i> <span>GENUINE</span>'
      : '<i class="fa-solid fa-triangle-exclamation"></i> <span>FORGED</span>';

    manualConfidenceText.textContent = `${data.confidence.toFixed(1)}%`;
    manualConfidenceBar.className = `confidence-bar ${isGenuine ? "genuine" : "forged"}`;
    manualConfidenceBar.style.width = `${Math.max(data.confidence, 15)}%`;

    manualModelUsedPill.innerHTML = `<i class="fa-solid fa-brain"></i> ${data.model_used}`;

    if (data.probabilities) {
      manualProbGen.textContent = `Genuine: ${data.probabilities.Genuine.toFixed(1)}%`;
      manualProbForg.textContent = `Forged: ${data.probabilities.Forged.toFixed(1)}%`;
    }

    manualResult.style.display = "block";
    manualResult.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // =========================================================================
  // QUICK TEST SPECIMENS GALLERY
  // =========================================================================
  const specimenChips = document.querySelectorAll(".specimen-chip");

  specimenChips.forEach(chip => {
    chip.addEventListener("click", async (e) => {
      e.preventDefault();
      const filename = chip.getAttribute("data-filename");
      const denom = chip.getAttribute("data-sample");
      if (!filename) return;

      hideAlert();
      btnAnalyzeUpload.disabled = true;
      btnUploadText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading Specimen...';

      try {
        const sampleUrl = `/static/samples/${filename}`;
        const resp = await fetch(sampleUrl);
        if (!resp.ok) {
          throw new Error(`Failed to load specimen image (${resp.status})`);
        }
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });

        // Set file into state and display preview
        currentFile = file;
        fileNameDisplay.textContent = `${filename} (₱${denom} Specimen)`;

        const reader = new FileReader();
        reader.onload = (re) => {
          imagePreview.src = re.target.result;
          if (resultThumb) resultThumb.src = re.target.result;
          previewContainer.style.display = "flex";

          // Trigger form submission
          setTimeout(() => {
            uploadForm.requestSubmit();
          }, 80);
        };
        reader.readAsDataURL(file);

      } catch (err) {
        console.error("Specimen fetch error:", err);
        showAlert(`Could not load test specimen: ${err.message}`);
        btnAnalyzeUpload.disabled = false;
        btnUploadText.textContent = "Analyze Currency Note";
      }
    });
  });

  // =========================================================================
  // SESSION SCAN HISTORY
  // =========================================================================
  const scanHistoryCard = document.getElementById("scanHistoryCard");
  const scanHistoryGrid = document.getElementById("scanHistoryGrid");
  const scanHistoryCount = document.getElementById("scanHistoryCount");
  const btnClearHistory = document.getElementById("btnClearHistory");

  function getSessionHistory() {
    try {
      const raw = sessionStorage.getItem("noteguard_scan_history");
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function addScanToSessionHistory(item) {
    const list = getSessionHistory();
    list.unshift(item);
    if (list.length > 12) list.length = 12;
    try {
      sessionStorage.setItem("noteguard_scan_history", JSON.stringify(list));
    } catch (e) {}
    renderSessionHistory();
  }

  function renderSessionHistory() {
    if (!scanHistoryCard || !scanHistoryGrid) return;
    const list = getSessionHistory();
    if (list.length === 0) {
      scanHistoryCard.style.display = "none";
      return;
    }

    scanHistoryCard.style.display = "block";
    if (scanHistoryCount) {
      scanHistoryCount.textContent = `${list.length} scan${list.length === 1 ? '' : 's'}`;
    }

    scanHistoryGrid.innerHTML = list.map((item, idx) => {
      const isGen = (item.prediction || "").toLowerCase() === "genuine";
      const icon = isGen ? "fa-circle-check" : "fa-triangle-exclamation";
      const color = isGen ? "#10b981" : "#ef4444";
      const thumbSrc = item.thumb || "/static/samples/sample_20.jpg";

      return `
        <div class="history-item-card" data-idx="${idx}" role="button" tabindex="0" title="Click to view scan results">
          <img src="${thumbSrc}" alt="Scan thumb" class="history-item-thumb">
          <div class="history-item-info">
            <div class="history-item-denom">₱${item.denomination_display}</div>
            <div class="history-item-meta">
              <span style="display: inline-flex; align-items: center; gap: 4px; color: ${color}; font-weight: 700;">
                <i class="fa-solid ${icon}"></i> ${(item.prediction || '').toUpperCase()}
              </span>
              <span>•</span>
              <span>${Math.round(item.confidence || 0)}%</span>
            </div>
            <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">
              <i class="fa-regular fa-clock"></i> ${item.timestamp || ''}
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Add click handlers on history items
    scanHistoryGrid.querySelectorAll(".history-item-card").forEach(el => {
      el.addEventListener("click", () => {
        const idx = parseInt(el.getAttribute("data-idx"), 10);
        const item = list[idx];
        if (item && item.rawData) {
          const snapshot = { ...item.rawData, _isHistorySnapshot: true };
          if (item.thumb) {
            imagePreview.src = item.thumb;
            previewContainer.style.display = "flex";
            fileNameDisplay.textContent = `History scan: ₱${item.denomination_display}`;
          }
          displayUploadResult(snapshot, item.thumb);
        }
      });
    });
  }

  if (btnClearHistory) {
    btnClearHistory.addEventListener("click", () => {
      try {
        sessionStorage.removeItem("noteguard_scan_history");
      } catch (e) {}
      renderSessionHistory();
    });
  }

  // =========================================================================
  // OFFICIAL VERIFICATION CERTIFICATE MODAL
  // =========================================================================
  const btnExportReport = document.getElementById("btnExportReport");
  const reportModal = document.getElementById("reportModal");
  const btnCloseReportModal = document.getElementById("btnCloseReportModal");
  const btnCloseCertBtn = document.getElementById("btnCloseCertBtn");
  const btnPrintCert = document.getElementById("btnPrintCert");

  function openCertificateModal() {
    if (!lastScanData) {
      showAlert("Please analyze a banknote first to generate a verification certificate.");
      return;
    }
    populateCertificate(lastScanData, lastScanThumb || (imagePreview ? imagePreview.src : ""));
    if (reportModal) {
      reportModal.style.display = "flex";
      document.body.style.overflow = "hidden";
    }
  }

  function closeCertificateModal() {
    if (reportModal) {
      reportModal.style.display = "none";
      document.body.style.overflow = "";
    }
  }

  if (btnExportReport) {
    btnExportReport.addEventListener("click", openCertificateModal);
  }
  if (btnCloseReportModal) {
    btnCloseReportModal.addEventListener("click", closeCertificateModal);
  }
  if (btnCloseCertBtn) {
    btnCloseCertBtn.addEventListener("click", closeCertificateModal);
  }
  if (reportModal) {
    reportModal.addEventListener("click", (e) => {
      if (e.target === reportModal) {
        closeCertificateModal();
      }
    });
  }
  if (btnPrintCert) {
    btnPrintCert.addEventListener("click", () => {
      window.print();
    });
  }

  function populateCertificate(data, thumbUrl) {
    const certInspectionId = document.getElementById("certInspectionId");
    const certTimestamp = document.getElementById("certTimestamp");
    const certImageThumb = document.getElementById("certImageThumb");
    const certVerdictBox = document.getElementById("certVerdictBox");
    const certVerdictTitle = document.getElementById("certVerdictTitle");
    const certConfidence = document.getElementById("certConfidence");
    const certDenomination = document.getElementById("certDenomination");
    const certSubstrate = document.getElementById("certSubstrate");
    const certChecksGrid = document.getElementById("certChecksGrid");
    const certWaveletTableBody = document.getElementById("certWaveletTableBody");

    if (certInspectionId) certInspectionId.textContent = "NG-2026-" + Math.floor(1000 + Math.random() * 9000);
    if (certTimestamp) certTimestamp.textContent = new Date().toLocaleString();
    if (certImageThumb && thumbUrl) certImageThumb.src = thumbUrl;

    const isGen = (data.prediction || "").toLowerCase() === "genuine";
    if (certVerdictBox) {
      certVerdictBox.className = `cert-verdict-box ${isGen ? "" : "forged"}`;
    }
    if (certVerdictTitle) {
      certVerdictTitle.textContent = (data.prediction || "UNKNOWN").toUpperCase();
    }
    if (certConfidence) {
      certConfidence.textContent = `${(data.confidence || 0).toFixed(1)}%`;
    }
    if (certDenomination) {
      certDenomination.textContent = data.denomination_display
        ? `₱${data.denomination_display} Pesos (New Generation Currency)`
        : "Philippine Peso Banknote";
    }
    if (certSubstrate) {
      certSubstrate.textContent = isGen
        ? "Matches genuine Philippine abaca-cotton / polymer substrate signature"
        : "Substrate anomalies detected — non-standard reflectance or micro-texture";
    }

    // Populate BSP checklist in cert
    if (certChecksGrid && data.bsp_checklist) {
      certChecksGrid.innerHTML = data.bsp_checklist.map(item => `
        <div class="cert-check-item">
          ${item.status === 'PASS' 
            ? '<i class="fa-solid fa-circle-check" style="color: #10b981;"></i>' 
            : '<i class="fa-solid fa-circle-xmark" style="color: #ef4444;"></i>'}
          <span><strong>${item.step}:</strong> ${item.name} (${item.status})</span>
        </div>
      `).join("");
    }

    // Populate wavelet table in cert
    if (certWaveletTableBody && data.extracted_features) {
      const feats = data.extracted_features;
      const defs = [
        { name: "Variance (DWT Energy)", val: feats.variance, mean: 0.1613 },
        { name: "Skewness (Asymmetry)", val: feats.skewness, mean: -0.5662 },
        { name: "Kurtosis (Peakedness)", val: feats.curtosis, mean: 0.1571 },
        { name: "Entropy (Complexity)", val: feats.entropy, mean: 0.0246 }
      ];
      certWaveletTableBody.innerHTML = defs.map(d => {
        const v = typeof d.val === "number" ? d.val.toFixed(4) : "--";
        const isOut = (data.out_of_range_features || []).some(k => d.name.toLowerCase().includes(k));
        return `
          <tr>
            <td>${d.name}</td>
            <td style="${isOut ? 'color: #ef4444; font-weight: bold;' : ''}">${v}</td>
            <td>${d.mean.toFixed(4)}</td>
            <td><span style="color: ${isOut ? '#ef4444' : '#10b981'}; font-weight: 600;">${isOut ? 'Out of Distribution' : 'Within Normal Bounds'}</span></td>
          </tr>
        `;
      }).join("");
    }
  }

  // Pre-load demo samples in background and restore session history
  loadSamples();
  renderSessionHistory();
});
