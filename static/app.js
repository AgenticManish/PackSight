/**
 * PackSight Frontend Application
 * Handles live camera feed with real-time quality heuristics,
 * multi-frame capture, image upload, side-by-side evidence inspection,
 * Legal Metrology compliance reporting, and scan history.
 */

(() => {
  // App state
  const state = {
    cameraStream: null,
    qualityInterval: null,
    capturedFrames: [], // Array of base64 data URLs
    selectedFiles: [], // Array of File objects for upload
    currentResult: null,
    activeImageKey: null,
    activeViewMode: "highlighted", // 'highlighted' | 'original' | 'split'
    activeFilter: "all", // 'all' | 'mandatory' | 'optional'
  };

  const FIELD_COLORS = {
    "MRP": "#00c4ff",
    "NET QUANTITY": "#00dc6e",
    "MANUFACTURER": "#ff9600",
    "MARKETED BY": "#ff5aaa",
    "CUSTOMER CARE": "#ff5050",
    "MFG / PACKING DATE": "#aa5aff",
    "USE BY / BEST BEFORE": "#be00ff",
    "LOT / BATCH": "#00ebeb",
    "COUNTRY OF ORIGIN": "#82be00",
    "LICENSE": "#3cb4ff",
    "INGREDIENTS": "#3cffc8",
    "MULTI UNIT PACKAGE": "#ffd200",
  };

  // Elements
  const el = {
    // Header
    btnQuickDemo: document.getElementById("btn-quick-demo"),
    btnToggleHistory: document.getElementById("btn-toggle-history"),

    // Tabs
    tabBtnCamera: document.getElementById("tab-btn-camera"),
    tabBtnUpload: document.getElementById("tab-btn-upload"),
    tabCamera: document.getElementById("tab-camera"),
    tabUpload: document.getElementById("tab-upload"),

    // Camera
    video: document.getElementById("camera-feed"),
    canvas: document.getElementById("camera-canvas"),
    btnStartCamera: document.getElementById("btn-start-camera"),
    btnCaptureFrame: document.getElementById("btn-capture-frame"),
    btnAddAnotherSideCam: document.getElementById("btn-add-another-side-cam"),
    btnScanCaptured: document.getElementById("btn-scan-captured"),
    capturedFramesList: document.getElementById("captured-frames-list"),
    frameCount: document.getElementById("frame-count"),
    cameraTip: document.getElementById("camera-tip"),
    qualitySharpness: document.getElementById("quality-sharpness"),
    qualityLighting: document.getElementById("quality-lighting"),
    qualityGlare: document.getElementById("quality-glare"),

    // Upload
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    uploadTrayBar: document.getElementById("upload-tray-bar"),
    btnAddMoreUpload: document.getElementById("btn-add-more-upload"),
    btnClearUpload: document.getElementById("btn-clear-upload"),
    uploadPreviewList: document.getElementById("upload-preview-list"),
    btnScanUploaded: document.getElementById("btn-scan-uploaded"),

    // Results container
    scanLoading: document.getElementById("scan-loading"),
    resultsEmpty: document.getElementById("results-empty"),
    resultsContainer: document.getElementById("results-container"),

    // Compliance Card
    complianceBadge: document.getElementById("compliance-badge"),
    productIdBadge: document.getElementById("product-id-badge"),
    productTitle: document.getElementById("product-title"),
    complianceMeta: document.getElementById("compliance-meta"),
    complianceRate: document.getElementById("compliance-rate"),
    statMandatory: document.getElementById("stat-mandatory"),
    statOptional: document.getElementById("stat-optional"),
    statDuration: document.getElementById("stat-duration"),
    nextIdText: document.getElementById("next-id-text"),

    // Visual Inspection Studio
    imageTabs: document.getElementById("image-tabs"),
    stageContainer: document.getElementById("stage-container"),
    frameOriginal: document.getElementById("frame-original"),
    frameHighlighted: document.getElementById("frame-highlighted"),
    imgOriginal: document.getElementById("img-original"),
    imgHighlighted: document.getElementById("img-highlighted"),
    viewModeBtns: document.querySelectorAll(".view-mode-btn"),

    // Declarations table
    declarationsTbody: document.getElementById("declarations-tbody"),
    filterBtns: document.querySelectorAll(".filter-btn"),
    rawJsonCode: document.getElementById("raw-json-code"),

    // History drawer
    historyDrawer: document.getElementById("history-drawer"),
    drawerBackdrop: document.getElementById("drawer-backdrop"),
    btnCloseHistory: document.getElementById("btn-close-history"),
    historyList: document.getElementById("history-list"),
  };

  // -------------------------------------------------------------------------
  // Initialization & Event Listeners
  // -------------------------------------------------------------------------

  function init() {
    setupTabs();
    setupCameraControls();
    setupUploadControls();
    setupViewModeToggles();
    setupFilters();
    setupHistoryDrawer();
    fetchNextProductId();

    // Quick demo button (runs Product 1 instantly)
    el.btnQuickDemo.addEventListener("click", () => runQuickDemo("product-1"));
  }

  async function fetchNextProductId() {
    try {
      const res = await fetch("/api/next-id");
      if (res.ok) {
        const data = await res.json();
        if (el.nextIdText && data.next_product_id) {
          el.nextIdText.textContent = data.next_product_id;
        }
      }
    } catch (e) {
      console.warn("Could not fetch next product ID:", e);
    }
  }

  // -------------------------------------------------------------------------
  // Tabs Navigation
  // -------------------------------------------------------------------------

  function setupTabs() {
    el.tabBtnCamera.addEventListener("click", () => {
      el.tabBtnCamera.classList.add("active");
      el.tabBtnUpload.classList.remove("active");
      el.tabCamera.classList.add("active");
      el.tabUpload.classList.remove("active");
    });

    el.tabBtnUpload.addEventListener("click", () => {
      el.tabBtnUpload.classList.add("active");
      el.tabBtnCamera.classList.remove("active");
      el.tabUpload.classList.add("active");
      el.tabCamera.classList.remove("active");
      stopCamera();
    });
  }

  // -------------------------------------------------------------------------
  // Camera & Live Quality Heuristics
  // -------------------------------------------------------------------------

  function setupCameraControls() {
    el.btnStartCamera.addEventListener("click", () => {
      if (state.cameraStream) {
        stopCamera();
      } else {
        startCamera();
      }
    });

    el.btnCaptureFrame.addEventListener("click", captureCurrentFrame);
    if (el.btnAddAnotherSideCam) {
      el.btnAddAnotherSideCam.addEventListener("click", () => {
        if (!state.cameraStream) {
          startCamera();
        }
        el.cameraTip.textContent = `Align next side (Side ${state.capturedFrames.length + 1}) within guide frame. Keep steady.`;
      });
    }
    el.btnScanCaptured.addEventListener("click", scanCapturedFrames);
  }

  async function startCamera() {
    try {
      el.cameraTip.textContent = "Requesting camera permissions...";
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      state.cameraStream = stream;
      el.video.srcObject = stream;
      el.btnStartCamera.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
        Stop Camera
      `;
      el.btnStartCamera.classList.replace("btn-primary", "btn-outline");
      el.btnCaptureFrame.disabled = false;
      el.cameraTip.textContent = "Align package text within guide frame. Keep steady.";

      // Begin quality monitoring interval
      state.qualityInterval = setInterval(checkLiveFrameQuality, 1400);
    } catch (err) {
      console.error("Camera access error:", err);
      el.cameraTip.textContent = `Camera access denied or unavailable: ${err.message}. Use file upload instead.`;
    }
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((track) => track.stop());
      state.cameraStream = null;
    }
    if (state.qualityInterval) {
      clearInterval(state.qualityInterval);
      state.qualityInterval = null;
    }
    el.video.srcObject = null;
    el.btnStartCamera.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
      Start Camera
    `;
    el.btnStartCamera.classList.replace("btn-outline", "btn-primary");
    el.btnCaptureFrame.disabled = true;
    resetQualityBanner();
    el.cameraTip.textContent = 'Click "Start Camera" to align and capture package faces';
  }

  async function checkLiveFrameQuality() {
    if (!state.cameraStream || el.video.videoWidth === 0) return;

    try {
      // Grab low-res frame for fast quality estimation
      const ctx = el.canvas.getContext("2d");
      const w = 480;
      const h = Math.round((el.video.videoHeight / el.video.videoWidth) * w);
      el.canvas.width = w;
      el.canvas.height = h;
      ctx.drawImage(el.video, 0, 0, w, h);
      const b64 = el.canvas.toDataURL("image/jpeg", 0.65);

      const res = await fetch("/api/quality-check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: b64 }),
      });

      if (!res.ok) return;
      const quality = await res.json();
      updateQualityBanner(quality);
    } catch (e) {
      console.warn("Quality check error:", e);
    }
  }

  function updateQualityBanner(q) {
    // Sharpness
    const sharpText = q.is_blurry ? "Blurry" : "Sharp";
    const sharpClass = q.is_blurry ? "bad" : "good";
    el.qualitySharpness.className = `quality-pill ${sharpClass}`;
    el.qualitySharpness.querySelector(".indicator-text").textContent = `Sharpness: ${sharpText} (${Math.round(q.blur_score)})`;

    // Lighting
    let lightText = "Optimal";
    let lightClass = "good";
    if (q.brightness < 45) {
      lightText = "Too Dark";
      lightClass = "bad";
    } else if (q.brightness > 220) {
      lightText = "Too Bright";
      lightClass = "warn";
    }
    el.qualityLighting.className = `quality-pill ${lightClass}`;
    el.qualityLighting.querySelector(".indicator-text").textContent = `Lighting: ${lightText}`;

    // Glare
    let glareText = "Low";
    let glareClass = "good";
    if (q.glare_percent > 8) {
      glareText = "High Glare";
      glareClass = "bad";
    } else if (q.glare_percent > 3) {
      glareText = "Moderate";
      glareClass = "warn";
    }
    el.qualityGlare.className = `quality-pill ${glareClass}`;
    el.qualityGlare.querySelector(".indicator-text").textContent = `Glare: ${glareText}`;

    // Tip
    if (q.recommendations && q.recommendations.length > 0) {
      el.cameraTip.textContent = q.recommendations[0];
    }
  }

  function resetQualityBanner() {
    el.qualitySharpness.className = "quality-pill";
    el.qualitySharpness.querySelector(".indicator-text").textContent = "Sharpness: --";
    el.qualityLighting.className = "quality-pill";
    el.qualityLighting.querySelector(".indicator-text").textContent = "Lighting: --";
    el.qualityGlare.className = "quality-pill";
    el.qualityGlare.querySelector(".indicator-text").textContent = "Glare: --";
  }

  function captureCurrentFrame() {
    if (!state.cameraStream || el.video.videoWidth === 0) return;

    // Full-resolution capture
    const ctx = el.canvas.getContext("2d");
    el.canvas.width = el.video.videoWidth;
    el.canvas.height = el.video.videoHeight;
    ctx.drawImage(el.video, 0, 0, el.video.videoWidth, el.video.videoHeight);
    const fullB64 = el.canvas.toDataURL("image/jpeg", 0.95);

    state.capturedFrames.push(fullB64);
    renderCapturedFrames();
  }

  function renderCapturedFrames() {
    const list = el.capturedFramesList;
    list.innerHTML = "";

    const count = state.capturedFrames.length;
    if (el.frameCount) el.frameCount.textContent = count;

    if (el.btnAddAnotherSideCam) {
      el.btnAddAnotherSideCam.style.display = count > 0 ? "inline-flex" : "none";
    }

    if (count === 0) {
      list.innerHTML = '<span class="empty-hint">No frames captured yet. Capture Front, Back, and Side faces for complete compliance.</span>';
      el.btnScanCaptured.disabled = true;
      el.btnScanCaptured.textContent = "Analyze Captured Angles";
      return;
    }

    state.capturedFrames.forEach((b64, idx) => {
      const wrapper = document.createElement("div");
      wrapper.className = "frame-thumbnail-wrapper";

      const img = document.createElement("img");
      img.src = b64;
      img.alt = `Side ${idx + 1}`;

      const label = document.createElement("span");
      label.className = "side-thumb-label";
      label.textContent = idx === 0 ? "Side 1 (Front)" : idx === 1 ? "Side 2 (Back)" : `Side ${idx + 1}`;

      const btnRemove = document.createElement("button");
      btnRemove.className = "btn-remove-thumb";
      btnRemove.innerHTML = "×";
      btnRemove.title = "Remove frame";
      btnRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        state.capturedFrames.splice(idx, 1);
        renderCapturedFrames();
      });

      wrapper.appendChild(img);
      wrapper.appendChild(label);
      wrapper.appendChild(btnRemove);
      list.appendChild(wrapper);
    });

    el.btnScanCaptured.disabled = false;
    el.btnScanCaptured.textContent = `Analyze ${count} Captured Side(s)`;
  }

  async function scanCapturedFrames() {
    if (state.capturedFrames.length === 0) return;
    stopCamera();
    showLoading();

    try {
      const res = await fetch("/api/scan-base64", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: state.capturedFrames }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Scan request failed.");
      }

      const result = await res.json();
      renderResults(result);
    } catch (err) {
      alert(`Scanning error: ${err.message}`);
      hideLoading();
    }
  }

  // -------------------------------------------------------------------------
  // File Upload Controls
  // -------------------------------------------------------------------------

  function setupUploadControls() {
    el.dropZone.addEventListener("click", () => el.fileInput.click());

    if (el.btnAddMoreUpload) {
      el.btnAddMoreUpload.addEventListener("click", () => el.fileInput.click());
    }
    if (el.btnClearUpload) {
      el.btnClearUpload.addEventListener("click", () => {
        state.selectedFiles = [];
        renderUploadPreviews();
      });
    }

    el.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.dropZone.classList.add("drag-over");
    });
    el.dropZone.addEventListener("dragleave", () => {
      el.dropZone.classList.remove("drag-over");
    });
    el.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      el.dropZone.classList.remove("drag-over");
      handleFilesSelected(Array.from(e.dataTransfer.files));
    });

    el.fileInput.addEventListener("change", (e) => {
      handleFilesSelected(Array.from(e.target.files));
    });

    el.btnScanUploaded.addEventListener("click", scanUploadedFiles);
  }

  function handleFilesSelected(files) {
    const valid = files.filter((f) => f.type.startsWith("image/"));
    if (valid.length === 0) return;

    state.selectedFiles.push(...valid);
    renderUploadPreviews();
  }

  function renderUploadPreviews() {
    const list = el.uploadPreviewList;
    list.innerHTML = "";

    const count = state.selectedFiles.length;
    if (el.uploadTrayBar) {
      el.uploadTrayBar.style.display = count > 0 ? "flex" : "none";
    }

    if (count === 0) {
      el.btnScanUploaded.disabled = true;
      el.btnScanUploaded.textContent = "Scan Uploaded Package";
      return;
    }

    state.selectedFiles.forEach((file, idx) => {
      const wrapper = document.createElement("div");
      wrapper.className = "frame-thumbnail-wrapper";

      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.alt = file.name;

      const label = document.createElement("span");
      label.className = "side-thumb-label";
      label.textContent = idx === 0 ? "Side 1 (Front)" : idx === 1 ? "Side 2 (Back)" : `Side ${idx + 1}`;

      const btnRemove = document.createElement("button");
      btnRemove.className = "btn-remove-thumb";
      btnRemove.innerHTML = "×";
      btnRemove.title = "Remove image";
      btnRemove.addEventListener("click", (e) => {
        e.stopPropagation();
        state.selectedFiles.splice(idx, 1);
        renderUploadPreviews();
      });

      wrapper.appendChild(img);
      wrapper.appendChild(label);
      wrapper.appendChild(btnRemove);
      list.appendChild(wrapper);
    });

    el.btnScanUploaded.disabled = false;
    el.btnScanUploaded.textContent = `Scan Uploaded Package (${count} Panels)`;
  }

  async function scanUploadedFiles() {
    if (state.selectedFiles.length === 0) return;
    showLoading();

    const formData = new FormData();
    state.selectedFiles.forEach((file) => {
      formData.append("files", file);
    });

    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Scan request failed.");
      }

      const result = await res.json();
      renderResults(result);
    } catch (err) {
      alert(`Scanning error: ${err.message}`);
      hideLoading();
    }
  }

  // -------------------------------------------------------------------------
  // Quick Demo Trigger
  // -------------------------------------------------------------------------

  async function runQuickDemo(productId) {
    showLoading();
    try {
      const res = await fetch(`/api/test-product/${productId}`, {
        method: "POST",
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Product demo scan failed.");
      }
      const result = await res.json();
      renderResults(result);
    } catch (err) {
      alert(`Quick Demo error: ${err.message}`);
      hideLoading();
    }
  }

  // -------------------------------------------------------------------------
  // Results Display & Visual Studio
  // -------------------------------------------------------------------------

  function showLoading() {
    el.resultsEmpty.style.display = "none";
    el.resultsContainer.style.display = "none";
    el.scanLoading.style.display = "flex";
  }

  function hideLoading() {
    el.scanLoading.style.display = "none";
  }

  function renderResults(result) {
    hideLoading();
    state.currentResult = result;
    el.resultsEmpty.style.display = "none";
    el.resultsContainer.style.display = "flex";

    const comp = result.compliance || {};
    const status = comp.overall_status || "NON_COMPLIANT";

    // 1. Compliance Card
    el.complianceBadge.textContent = status.replace("_", " ");
    el.complianceBadge.className = `compliance-badge ${status.toLowerCase().replace("_", "-")}`;
    const scoreCircle = document.querySelector(".compliance-score-circle");
    if (scoreCircle) {
      scoreCircle.className = `compliance-score-circle ${status.toLowerCase().replace("_", "-")}`;
    }
    if (el.productIdBadge) {
      el.productIdBadge.textContent = result.product || "PS-2026-XXXX";
    }
    const brandDisplay = result.brand_name || result.product_name || (result.product && !result.product.startsWith("PS-") ? result.product : "Unknown Product");
    el.productTitle.textContent = brandDisplay;

    const sideCount = Object.keys(result.images || {}).length;
    el.complianceMeta.textContent = `Scan ID: ${result.scan_id ? result.scan_id.substring(0, 16) : "--"} | ${sideCount} Side(s) Inspected | Legal Metrology Rules, 2011`;

    const rate = comp.compliance_rate_percent !== undefined ? comp.compliance_rate_percent : 0;
    el.complianceRate.textContent = `${Math.round(rate)}%`;

    const compliantMand = comp.mandatory_fields_compliant !== undefined ? comp.mandatory_fields_compliant : (comp.mandatory_fields_present || 0);
    const reviewMand = comp.mandatory_fields_review || 0;
    el.statMandatory.textContent = reviewMand > 0
      ? `${compliantMand} / ${comp.mandatory_fields_total || 9} Verified (${reviewMand} Review)`
      : `${compliantMand} / ${comp.mandatory_fields_total || 9} Present`;
    
    // Optional count
    const optionalCount = Object.values(result.fields || {}).filter(
      (f) => !["MRP", "NET QUANTITY", "MANUFACTURER", "MARKETED BY", "CUSTOMER CARE", "MFG / PACKING DATE", "USE BY / BEST BEFORE", "LOT / BATCH", "COUNTRY OF ORIGIN"].includes(f.field)
    ).length;
    el.statOptional.textContent = `${optionalCount} Detected`;
    el.statDuration.textContent = `${result.duration_seconds || "--"} s`;

    // 2. Visual Inspection Studio Setup
    setupInspectionStudio(result.images || {});

    // 3. Declarations Table
    renderDeclarationsTable(comp.field_breakdown || {}, result.fields || {});

    // 4. Raw JSON
    el.rawJsonCode.textContent = JSON.stringify(result, null, 2);

    // Refresh dynamic target ID for next live product scan
    fetchNextProductId();
  }

  function setupInspectionStudio(images) {
    const tabsContainer = el.imageTabs;
    tabsContainer.innerHTML = "";

    const imageKeys = Object.keys(images);
    if (imageKeys.length === 0) return;

    state.activeImageKey = imageKeys[0];

    imageKeys.forEach((key, idx) => {
      const btn = document.createElement("button");
      btn.className = `image-tab-btn ${key === state.activeImageKey ? "active" : ""}`;
      const cleanName = key.replace(/\.(jpeg|jpg|png|webp)$/i, "");
      btn.textContent = idx === 0 && cleanName.toLowerCase().includes("front") ? "Side 1 (Front)" :
                        idx === 1 && cleanName.toLowerCase().includes("back") ? "Side 2 (Back)" :
                        `Side ${idx + 1} (${cleanName})`;
      btn.addEventListener("click", () => {
        state.activeImageKey = key;
        document.querySelectorAll(".image-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        updateStageImages();
      });
      tabsContainer.appendChild(btn);
    });

    updateStageImages();
  }

  function updateStageImages() {
    if (!state.currentResult || !state.activeImageKey) return;
    const imgData = state.currentResult.images[state.activeImageKey];
    if (!imgData) return;

    el.imgOriginal.src = imgData.original_url || "";
    el.imgHighlighted.src = imgData.highlighted_url || "";

    applyViewMode(state.activeViewMode);
  }

  function setupViewModeToggles() {
    el.viewModeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        el.viewModeBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeViewMode = btn.dataset.mode;
        applyViewMode(state.activeViewMode);
      });
    });
  }

  function applyViewMode(mode) {
    el.stageContainer.className = `stage-container mode-${mode}`;

    if (mode === "highlighted") {
      el.frameOriginal.style.display = "none";
      el.frameHighlighted.style.display = "flex";
    } else if (mode === "original") {
      el.frameOriginal.style.display = "flex";
      el.frameHighlighted.style.display = "none";
    } else if (mode === "split") {
      el.frameOriginal.style.display = "flex";
      el.frameHighlighted.style.display = "flex";
    }
  }

  // -------------------------------------------------------------------------
  // Declarations Table & Filters
  // -------------------------------------------------------------------------

  function setupFilters() {
    el.filterBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        el.filterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeFilter = btn.dataset.filter;
        if (state.currentResult) {
          const comp = state.currentResult.compliance || {};
          renderDeclarationsTable(comp.field_breakdown || {}, state.currentResult.fields || {});
        }
      });
    });
  }

  function renderDeclarationsTable(breakdown, fields) {
    const tbody = el.declarationsTbody;
    tbody.innerHTML = "";

    const fieldEntries = Object.entries(breakdown);
    if (fieldEntries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty-hint">No field evaluations found.</td></tr>';
      return;
    }

    fieldEntries.forEach(([fieldName, item]) => {
      const isMandatory = item.mandatory !== false;

      // Filter check
      if (state.activeFilter === "mandatory" && !isMandatory) return;
      if (state.activeFilter === "optional" && isMandatory) return;

      const tr = document.createElement("tr");

      // 1. Field Name with Color Dot
      const color = FIELD_COLORS[fieldName] || "#58a6ff";
      const nameTd = document.createElement("td");
      nameTd.innerHTML = `
        <div class="field-name-cell">
          <span class="field-indicator-dot" style="background-color: ${color}; box-shadow: 0 0 6px ${color};"></span>
          <span>${fieldName}</span>
        </div>
      `;

      // 2. Category
      const catTd = document.createElement("td");
      catTd.innerHTML = isMandatory
        ? '<span class="badge-tag mandatory">Mandatory</span>'
        : '<span class="badge-tag optional">Optional</span>';

      // 3. Detection Status
      const detTd = document.createElement("td");
      const detStatus = item.detection_status || (item.value && item.confidence >= 30.0 ? "DETECTED" : item.value ? "REVIEW" : "MISSING");
      detTd.innerHTML = `<span class="status-tag ${detStatus.toLowerCase()}">${detStatus}</span>`;

      // 4. Compliance Status
      const statusTd = document.createElement("td");
      const status = item.status || "MISSING";
      const statusClass = status.toLowerCase().replace("_", "-");
      statusTd.innerHTML = `<span class="status-tag ${statusClass}">${status.replace("_", " ")}</span>`;

      // 5. Extracted Value
      const valTd = document.createElement("td");
      valTd.className = item.value ? "value-cell" : "value-cell empty";
      valTd.textContent = item.value || "Not Detected / Missing";

      // 6. Confidence Meter
      const confTd = document.createElement("td");
      const conf = Math.round(item.confidence || 0);
      confTd.innerHTML = `
        <div class="confidence-bar-wrapper">
          <div class="confidence-track">
            <div class="confidence-fill" style="width: ${conf}%;"></div>
          </div>
          <span class="confidence-num">${conf}%</span>
        </div>
      `;

      // 7. Audit Reason
      const reasonTd = document.createElement("td");
      const reasonClass = item.status === "REVIEW" ? "warning" : item.status === "NON_COMPLIANT" || item.status === "MISSING" ? "error" : "success";
      reasonTd.className = `reason-cell ${reasonClass}`;
      reasonTd.textContent = item.reason || item.detection_reason || (status === "COMPLIANT" ? "Verified" : status === "REVIEW" ? "Requires visual review" : "Missing mandatory declaration");

      // 8. Source Panel
      const srcTd = document.createElement("td");
      srcTd.style.color = "var(--text-muted)";
      srcTd.style.fontSize = "0.78rem";
      srcTd.textContent = item.source_image || fields[fieldName]?.source_image || "—";

      tr.appendChild(nameTd);
      tr.appendChild(catTd);
      tr.appendChild(detTd);
      tr.appendChild(statusTd);
      tr.appendChild(valTd);
      tr.appendChild(confTd);
      tr.appendChild(reasonTd);
      tr.appendChild(srcTd);

      tbody.appendChild(tr);
    });
  }

  // -------------------------------------------------------------------------
  // History Drawer
  // -------------------------------------------------------------------------

  function setupHistoryDrawer() {
    el.btnToggleHistory.addEventListener("click", openHistoryDrawer);
    el.btnCloseHistory.addEventListener("click", closeHistoryDrawer);
    el.drawerBackdrop.addEventListener("click", closeHistoryDrawer);
  }

  async function openHistoryDrawer() {
    el.historyDrawer.classList.add("open");
    el.drawerBackdrop.classList.add("open");
    el.historyList.innerHTML = '<div class="empty-hint">Loading history records...</div>';

    try {
      const res = await fetch("/api/history");
      const records = await res.json();
      renderHistoryList(records);
    } catch (e) {
      el.historyList.innerHTML = `<div class="empty-hint">Failed to load history: ${e.message}</div>`;
    }
  }

  function closeHistoryDrawer() {
    el.historyDrawer.classList.remove("open");
    el.drawerBackdrop.classList.remove("open");
  }

  function renderHistoryList(records) {
    const list = el.historyList;
    list.innerHTML = "";

    if (!records || records.length === 0) {
      list.innerHTML = '<div class="empty-hint">No scan history records yet.</div>';
      return;
    }

    records.forEach((rec) => {
      const card = document.createElement("div");
      card.className = "history-item";

      const timeStr = rec.scanned_at ? new Date(rec.scanned_at).toLocaleString() : "--";
      const status = rec.overall_status || "UNKNOWN";
      const prodId = rec.product_id || rec.product || "Package";
      const brand = rec.brand_name && rec.brand_name !== "Unknown Product" && rec.brand_name !== prodId
        ? `${rec.brand_name} · ${prodId}`
        : prodId;
      const sidesStr = rec.image_count ? `${rec.image_count} Side${rec.image_count > 1 ? "s" : ""}` : "Package";

      card.innerHTML = `
        <div class="history-item-top">
          <span class="history-item-name">${brand}</span>
          <span class="status-tag ${status.toLowerCase().replace("_", "-")}">${status.replace("_", " ")}</span>
        </div>
        <div class="history-item-stat">Compliance: ${Math.round(rec.compliance_rate_percent || 0)}% (${rec.mandatory_fields_present || 0}/${rec.mandatory_fields_total || 9}) · ${sidesStr}</div>
        <div class="history-item-time">${timeStr} · ${rec.duration_seconds || "--"}s</div>
      `;

      card.addEventListener("click", async () => {
        closeHistoryDrawer();
        const targetId = rec.product_id || rec.product;
        if (targetId) {
          showLoading();
          try {
            const productRes = await fetch(`/api/product/${targetId}`);
            if (productRes.ok) {
              const data = await productRes.json();
              renderResults(data);
            } else if (rec.scan_id) {
              const scanRes = await fetch(`/api/scan/${rec.scan_id}`);
              if (scanRes.ok) {
                const data = await scanRes.json();
                renderResults(data);
              } else {
                runQuickDemo(targetId);
              }
            } else {
              runQuickDemo(targetId);
            }
          } catch {
            runQuickDemo(targetId);
          }
        }
      });

      list.appendChild(card);
    });
  }

  // Start app on DOM load
  document.addEventListener("DOMContentLoaded", init);
})();
