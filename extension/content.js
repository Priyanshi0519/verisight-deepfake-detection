/**
 * VeriSight - Content Script (v2.2)
 * Robust viewport-aware automatic image scanning with:
 * - Immediate ON/OFF state synchronization and DOM badge cleanup
 * - Zero lingering "Scanning..." badges on network errors or timeouts
 * - Deep scroll/resize anchoring with capturing listeners
 * - Productivity/Chat site exclusion (ChatGPT, Claude, GitHub, etc.)
 * - Dynamic generation cache-bypassing (thispersondoesnotexist.com)
 */

(() => {
  // Configuration defaults
  let config = {
    autoScanEnabled: true,
    dwellTimeMs: 500, // 0.5s network buffer debounce
    minImageDimension: 120 // Ignore tiny icons, badges, avatars, and UI glyphs
  };

  // Local state caches
  const imageCache = new Map(); // url -> prediction object
  const pendingTimers = new Map(); // img element -> setTimeout id
  const trackedImages = new WeakSet();
  const activeBadges = new Map(); // img element -> badge element
  const scanTimeouts = new Map(); // img element -> watchdog setTimeout id

  // 0. Domain & Environment Verification
  const hostname = window.location.hostname.toLowerCase();
  const isExcludedDomain = (
    hostname.includes("chatgpt.com") ||
    hostname.includes("claude.ai") ||
    hostname.includes("github.com") ||
    hostname.includes("overleaf.com") ||
    hostname.includes("mail.google.com")
  );

  // Stop execution immediately on excluded sites (ChatGPT, Claude, etc.)
  if (isExcludedDomain) {
    return;
  }

  // Safety check to prevent "Extension context invalidated" on extension reloads
  function isExtensionValid() {
    return typeof chrome !== "undefined" && Boolean(chrome?.runtime?.id);
  }

  function sendExtensionMessage(payload, callback) {
    if (!isExtensionValid()) return;
    try {
      chrome.runtime.sendMessage(payload, (response) => {
        if (!isExtensionValid() || chrome.runtime?.lastError) return;
        if (callback) callback(response);
      });
    } catch (e) {
      // Extension reloaded in developer mode; safely exit
    }
  }

  // Load user preferences from storage
  if (isExtensionValid()) {
    try {
      chrome.storage?.local?.get(["autoScanEnabled", "dwellTimeMs"], (items) => {
        if (!isExtensionValid() || chrome.runtime?.lastError) return;
        if (items && items.autoScanEnabled !== undefined) {
          setAutoScanState(items.autoScanEnabled);
        }
        if (items && items.dwellTimeMs !== undefined) {
          config.dwellTimeMs = items.dwellTimeMs;
        }
      });

      chrome.storage?.onChanged?.addListener((changes) => {
        if (!isExtensionValid()) return;
        if (changes.autoScanEnabled) {
          setAutoScanState(changes.autoScanEnabled.newValue);
        }
        if (changes.dwellTimeMs) {
          config.dwellTimeMs = changes.dwellTimeMs.newValue;
        }
      });
    } catch (e) {}
  }

  function setAutoScanState(enabled) {
    config.autoScanEnabled = Boolean(enabled);
    if (!config.autoScanEnabled) {
      // 1. Cancel all pending dwell-time timers
      pendingTimers.forEach((timerId) => clearTimeout(timerId));
      pendingTimers.clear();

      // 2. Clear scan watchdog timers
      scanTimeouts.forEach((timerId) => clearTimeout(timerId));
      scanTimeouts.clear();

      // 3. Clear batch queue
      if (batchTimeout) {
        clearTimeout(batchTimeout);
        batchTimeout = null;
      }
      batchQueue = [];

      // 4. Destroy ALL active badges from the DOM immediately
      const imgs = Array.from(activeBadges.keys());
      imgs.forEach((img) => removeBadge(img));
      activeBadges.clear();
    } else {
      // Re-scan visible DOM
      scanPageImages();
    }
  }

  // 1. Intersection Observer for Viewport Tracking
  const observer = new IntersectionObserver((entries) => {
    if (!config.autoScanEnabled || isExcludedDomain) return;

    entries.forEach((entry) => {
      const img = entry.target;

      if (entry.isIntersecting) {
        // Image entered viewport: start 0.5s dwell-time buffer
        if (!pendingTimers.has(img) && !activeBadges.has(img)) {
          const timerId = setTimeout(() => {
            pendingTimers.delete(img);
            if (config.autoScanEnabled && isEligibleImage(img)) {
              queueImageForInspection(img);
            }
          }, config.dwellTimeMs);

          pendingTimers.set(img, timerId);
        }
      } else {
        // Image left viewport before dwell time: cancel request to avoid network spam
        if (pendingTimers.has(img)) {
          clearTimeout(pendingTimers.get(img));
          pendingTimers.delete(img);
        }
      }
    });
  }, {
    rootMargin: "60px",
    threshold: 0.25
  });

  function isEligibleImage(img) {
    if (!config.autoScanEnabled || isExcludedDomain) return false;
    if (!img || !img.isConnected) return false;

    const src = img.currentSrc || img.getAttribute("data-src") || img.src;
    if (!src) return false;
    if (src.startsWith("data:image/svg") || src.endsWith(".svg")) return false;
    // Ignore 1x1 transparent spacer gifs or tiny tracking pixels
    if (src.startsWith("data:") && src.length < 120) return false;

    // Filter out navigation bars, header bars, menus, and tabs
    if (img.closest("nav, header, [role='menu'], [role='tab']")) {
      return false;
    }

    // Only filter out actual <button> elements if the image is a small icon inside it
    const buttonParent = img.closest("button");
    if (buttonParent) {
      const bRect = img.getBoundingClientRect();
      if (bRect.width < 100 || bRect.height < 100) return false;
    }

    // Filter out obvious icon/avatar classes
    const classString = (img.className || "").toLowerCase() + " " + (img.id || "").toLowerCase();
    if (classString.includes("avatar") || classString.includes("emoji")) {
      return false;
    }

    // Dimension check: Allow standard search thumbnails (>= 80px)
    const rect = img.getBoundingClientRect();
    const width = rect.width || img.naturalWidth || 0;
    const height = rect.height || img.naturalHeight || 0;
    return (width >= 80 && height >= 80) || (width * height >= 7500);
  }

  // 2. Queue and Batching System
  let batchQueue = [];
  let batchTimeout = null;

  function getBase64FromImage(img) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = Math.min(img.naturalWidth || img.width || 224, 640);
      canvas.height = Math.min(img.naturalHeight || img.height || 224, 640);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.85);
    } catch (e) {
      return null;
    }
  }

  function queueImageForInspection(img) {
    if (!config.autoScanEnabled) return;
    const src = img.currentSrc || img.getAttribute("data-src") || img.src;
    if (!src) return;
    if (src.startsWith("data:") && src.length < 120) return;

    const isDataUri = src.startsWith("data:image/");
    const isDynamic = src.includes("thispersondoesnotexist.com") || src.startsWith("blob:") || isDataUri;

    // Bypass cache completely for dynamic generation endpoints where URL stays identical on reload
    if (!isDynamic && imageCache.has(src)) {
      renderBadge(img, imageCache.get(src));
      return;
    }

    // Render "Scanning..." badge
    renderScanningBadge(img);

    // Direct Base64 bypass for inline search thumbnails (e.g. Google Images data:image/jpeg;base64)
    if (isDataUri) {
      sendExtensionMessage({ action: "CHECK_IMAGE", imageBase64: src }, (response) => {
        if (!config.autoScanEnabled) {
          removeBadge(img);
          return;
        }
        if (response && response.success && response.data && !response.data.error) {
          renderBadge(img, response.data);
        } else {
          removeBadge(img);
        }
      });
      return;
    }

    // Dynamic endpoints (e.g., thispersondoesnotexist.com): attempt canvas capture or append timestamp
    if (isDynamic) {
      const b64 = getBase64FromImage(img);
      if (b64) {
        sendExtensionMessage({ action: "CHECK_IMAGE", imageBase64: b64 }, (response) => {
          if (!config.autoScanEnabled) {
            removeBadge(img);
            return;
          }
          if (response && response.success && response.data && !response.data.error) {
            renderBadge(img, response.data);
          } else {
            removeBadge(img);
          }
        });
        return;
      }
    }

    // For dynamic URLs where canvas was blocked by CORS, add a query cache-buster so backend fetches fresh
    let requestUrl = src;
    if (isDynamic && src.includes("http")) {
      const sep = src.includes("?") ? "&" : "?";
      requestUrl = src + sep + "_cb=" + Date.now();
    }

    batchQueue.push({ img, src, requestUrl });

    if (!batchTimeout) {
      batchTimeout = setTimeout(processBatchQueue, 200);
    }
  }

  function processBatchQueue() {
    batchTimeout = null;
    if (batchQueue.length === 0) return;

    // Process in smooth chunks of 4 images so CPU finishes each chunk fast
    const currentBatch = batchQueue.splice(0, 4);

    // If more images remain in queue, schedule the next batch
    if (batchQueue.length > 0) {
      batchTimeout = setTimeout(processBatchQueue, 300);
    }

    if (!config.autoScanEnabled) {
      currentBatch.forEach((item) => removeBadge(item.img));
      return;
    }

    const urlMap = new Map();
    currentBatch.forEach((item) => {
      const u = item.requestUrl || item.src;
      if (!urlMap.has(u)) urlMap.set(u, []);
      urlMap.get(u).push(item.img);
    });

    const urls = Array.from(urlMap.keys());

    sendExtensionMessage({ action: "CHECK_BATCH", urls }, (response) => {
      if (!config.autoScanEnabled) {
        currentBatch.forEach((item) => removeBadge(item.img));
        return;
      }

      if (response && response.success && response.data && response.data.results) {
        const results = response.data.results;
        for (const [url, result] of Object.entries(results)) {
          const imgs = urlMap.get(url) || [];
          if (result && !result.error) {
            if (!url.includes("thispersondoesnotexist.com")) {
              imageCache.set(url, result);
            }
            imgs.forEach((img) => renderBadge(img, result));
          } else {
            // Display an informative error badge rather than silently vanishing!
            const errMsg = (result && result.error) ? result.error : "Scan failed";
            imgs.forEach((img) => renderErrorBadge(img, "⚠️ Error", errMsg));
          }
        }
      } else {
        // Backend offline or unreachable: Inform user clearly instead of vanishing!
        currentBatch.forEach((item) => renderErrorBadge(item.img, "⚠️ Server Offline", "Make sure python src/inference/app.py is running"));
      }
    });
  }

  function renderErrorBadge(img, label, reason) {
    if (scanTimeouts.has(img)) {
      clearTimeout(scanTimeouts.get(img));
      scanTimeouts.delete(img);
    }
    if (!config.autoScanEnabled) {
      removeBadge(img);
      return;
    }
    const badge = createOrGetBadgeWrapper(img);
    badge.innerHTML = `
      <div class="verisight-badge verisight-error" style="background: rgba(239, 68, 68, 0.25); border: 1px solid rgba(239, 68, 68, 0.6); color: #fca5a5; font-size: 10px; cursor: help;" title="${reason}">
        <span>${label}</span>
      </div>
    `;
    updateBadgePosition(img, badge);

    // Auto-clear error notice after 7 seconds
    setTimeout(() => {
      if (activeBadges.has(img)) {
        const b = activeBadges.get(img);
        if (b && b.querySelector(".verisight-error")) {
          removeBadge(img);
        }
      }
    }, 7000);
  }

  // 3. Badge Overlay Rendering & Anchoring
  function renderScanningBadge(img) {
    if (!config.autoScanEnabled) return;
    const badge = createOrGetBadgeWrapper(img);
    badge.innerHTML = `
      <div class="verisight-badge verisight-scanning">
        <span class="verisight-pulse-dot"></span>
        <span>Scanning...</span>
      </div>
    `;
    updateBadgePosition(img, badge);
  }

  function renderBadge(img, result) {
    if (scanTimeouts.has(img)) {
      clearTimeout(scanTimeouts.get(img));
      scanTimeouts.delete(img);
    }

    if (!config.autoScanEnabled || !result || result.error) {
      removeBadge(img);
      return;
    }

    const badge = createOrGetBadgeWrapper(img);
    const isAi = result.is_ai;
    const confidence = result.confidence_pct || Math.round((result.ai_probability || 0) * 100);

    let badgeClass = isAi ? "verisight-ai" : "verisight-real";
    let badgeText = isAi ? `🔴 AI (${confidence}%)` : `🟢 Real (${confidence}%)`;

    if (result.verdict === "Deepfake") {
      badgeClass = "verisight-deepfake";
      badgeText = `🔶 Deepfake (${confidence}%)`;
    }

    badge.innerHTML = `
      <div class="verisight-badge ${badgeClass}">
        <span>${badgeText}</span>
      </div>
    `;

    // Click handler to display forensic detail card
    const badgePill = badge.querySelector(".verisight-badge");
    if (badgePill) {
      badgePill.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        toggleDetailCard(img, badge, result);
      });
    }

    updateBadgePosition(img, badge);
  }

  function toggleDetailCard(img, badge, result) {
    const existingCard = badge.querySelector(".verisight-modal-card");
    if (existingCard) {
      existingCard.remove();
      return;
    }

    const aiProb = Math.round((result.ai_probability || 0) * 100);
    const spatialProb = Math.round((result.spatial_ai_probability || 0) * 100);
    const spectralProb = Math.round((result.spectral_ai_probability || 0) * 100);

    const card = document.createElement("div");
    card.className = "verisight-modal-card";
    card.innerHTML = `
      <div class="verisight-card-header">
        <span>VeriSight Forensics</span>
        <span class="verisight-close-btn">&times;</span>
      </div>
      <div><strong>Verdict:</strong> ${result.label} (${result.confidence_pct}%)</div>
      
      <div class="verisight-progress-bar">
        <div class="verisight-progress-fill" style="width: ${aiProb}%; background: ${result.is_ai ? '#ef4444' : '#10b981'};"></div>
      </div>
      
      <div class="verisight-metric-row">
        <span>Spatial (RGB Texture):</span>
        <strong>${spatialProb}% AI</strong>
      </div>
      <div class="verisight-metric-row">
        <span>Spectral (2D FFT):</span>
        <strong>${spectralProb}% AI</strong>
      </div>
      <div class="verisight-metric-row">
        <span>Inference Time:</span>
        <span>${result.processing_time_ms || 30} ms</span>
      </div>
      
      <div class="verisight-disclaimer">
        * Probabilistic indicator combining deep spatial cues & Fourier spectral analysis. Not definitive proof.
      </div>
    `;

    card.querySelector(".verisight-close-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      card.remove();
    });

    badge.appendChild(card);
  }

  function cleanupOrphanedBadges() {
    activeBadges.forEach((badge, img) => {
      if (!img.isConnected) {
        removeBadge(img);
      }
    });
  }

  function createOrGetBadgeWrapper(img) {
    cleanupOrphanedBadges();

    if (activeBadges.has(img)) {
      return activeBadges.get(img);
    }

    // Deduplicate: remove any existing badge occupying nearly the same coordinate tile
    const rect = img.getBoundingClientRect();
    for (const [otherImg, existingBadge] of activeBadges.entries()) {
      const oRect = otherImg.getBoundingClientRect();
      if (Math.abs(rect.top - oRect.top) < 15 && Math.abs(rect.left - oRect.left) < 15) {
        removeBadge(otherImg);
        break;
      }
    }

    const wrapper = document.createElement("div");
    wrapper.className = "verisight-badge-wrapper";
    document.body.appendChild(wrapper);
    activeBadges.set(img, wrapper);

    // Attach ResizeObserver to keep badge synchronized with layout shifts
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        if (activeBadges.has(img)) {
          updateBadgePosition(img, wrapper);
        }
      });
      ro.observe(img);
    }

    return wrapper;
  }

  function removeBadge(img) {
    if (scanTimeouts.has(img)) {
      clearTimeout(scanTimeouts.get(img));
      scanTimeouts.delete(img);
    }
    if (activeBadges.has(img)) {
      try {
        activeBadges.get(img).remove();
      } catch (e) {}
      activeBadges.delete(img);
    }
  }

  function updateBadgePosition(img, badge) {
    if (!img.isConnected) {
      removeBadge(img);
      return;
    }
    const rect = img.getBoundingClientRect();
    // If element is hidden, collapsed, or scrolled completely out of view, hide badge
    if (
      rect.width === 0 ||
      rect.height === 0 ||
      rect.bottom <= 0 ||
      rect.top >= window.innerHeight ||
      rect.right <= 0 ||
      rect.left >= window.innerWidth
    ) {
      badge.style.display = "none";
      return;
    }
    badge.style.display = "block";
    badge.style.top = `${rect.top + window.scrollY + 6}px`;
    badge.style.left = `${rect.right + window.scrollX - 110}px`;
  }

  // 4. Global Scroll & Resize Synchronization (Capture phase for all internal scroll containers)
  document.addEventListener("scroll", () => {
    activeBadges.forEach((badge, img) => updateBadgePosition(img, badge));
  }, { passive: true, capture: true });

  window.addEventListener("resize", () => {
    activeBadges.forEach((badge, img) => updateBadgePosition(img, badge));
  }, { passive: true });

  // 5. Scan DOM for Images with SPA source-change detection
  function attachImageListeners(img) {
    if (trackedImages.has(img)) return;
    trackedImages.add(img);

    const initialSrc = img.currentSrc || img.getAttribute("data-src") || img.src || "";
    img.dataset.verisightLastSrc = initialSrc;

    // Detect when Single Page Apps (Kaggle, Twitter, Google) change the image src
    img.addEventListener("load", () => {
      const currentSrc = img.currentSrc || img.getAttribute("data-src") || img.src || "";
      if (img.dataset.verisightLastSrc && img.dataset.verisightLastSrc !== currentSrc) {
        img.dataset.verisightLastSrc = currentSrc;
        removeBadge(img);
        if (config.autoScanEnabled && isEligibleImage(img)) {
          queueImageForInspection(img);
        }
      } else if (activeBadges.has(img)) {
        updateBadgePosition(img, activeBadges.get(img));
      }
    });

    observer.observe(img);
  }

  function scanPageImages() {
    if (!config.autoScanEnabled || isExcludedDomain) return;

    const images = document.querySelectorAll("img");
    images.forEach(attachImageListeners);
  }

  // Monitor dynamic DOM additions & SPA attribute changes
  const mutationObserver = new MutationObserver((mutations) => {
    if (!config.autoScanEnabled || isExcludedDomain) return;

    let hasDomChanges = false;
    for (const m of mutations) {
      if (m.type === "attributes" && m.target && m.target.tagName === "IMG") {
        const img = m.target;
        const currentSrc = img.currentSrc || img.getAttribute("data-src") || img.src || "";
        if (img.dataset.verisightLastSrc && img.dataset.verisightLastSrc !== currentSrc) {
          img.dataset.verisightLastSrc = currentSrc;
          removeBadge(img);
          if (isEligibleImage(img)) {
            queueImageForInspection(img);
          }
        }
      } else if (m.type === "childList") {
        hasDomChanges = true;
      }
    }

    if (hasDomChanges) {
      scanPageImages();
    }
    // Also update positions of any moving badges
    activeBadges.forEach((badge, img) => updateBadgePosition(img, badge));
  });

  mutationObserver.observe(document.body, { 
    childList: true, 
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "data-src"]
  });

  // Initial scan if allowed
  if (!isExcludedDomain) {
    scanPageImages();
  }

  // 6. Handle Context Menu / Single Image Verification
  chrome.runtime.onMessage?.addListener((msg) => {
    if (msg.action === "VERIFY_SINGLE_IMAGE" && msg.srcUrl) {
      const allImgs = Array.from(document.querySelectorAll("img"));
      const targetImg = allImgs.find((i) => (i.currentSrc || i.src) === msg.srcUrl) || allImgs[0];
      if (targetImg) {
        queueImageForInspection(targetImg);
      }
    }
  });

  console.log("VeriSight content script v2.2 active: Dynamic anchoring, fast OFF toggle & auto-cleanup enabled.");
})();
