/**
 * VeriSight Popup Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  const statusBadge = document.getElementById("server-status");
  const statusText = document.getElementById("status-text");
  const toggleAutoScan = document.getElementById("toggle-autoscan");
  const dwellSlider = document.getElementById("dwell-slider");
  const dwellVal = document.getElementById("dwell-val");
  const manualUrlInput = document.getElementById("manual-url");
  const checkBtn = document.getElementById("check-btn");
  const resultBox = document.getElementById("manual-result");
  const resLabel = document.getElementById("res-label");
  const resConf = document.getElementById("res-conf");
  const resSpatial = document.getElementById("res-spatial");
  const resSpectral = document.getElementById("res-spectral");

  // 1. Check Backend Server Health
  function checkHealth() {
    chrome.runtime.sendMessage({ action: "HEALTH_CHECK" }, (response) => {
      if (response && response.success && response.data) {
        statusBadge.className = "status-badge status-online";
        statusText.textContent = "Online";
      } else {
        statusBadge.className = "status-badge status-offline";
        statusText.textContent = "Offline";
      }
    });
  }
  checkHealth();

  // 2. Load User Preferences
  chrome.storage.local.get(["autoScanEnabled", "dwellTimeMs"], (items) => {
    if (items.autoScanEnabled !== undefined) {
      toggleAutoScan.checked = items.autoScanEnabled;
    }
    if (items.dwellTimeMs !== undefined) {
      dwellSlider.value = items.dwellTimeMs;
      dwellVal.textContent = `${items.dwellTimeMs}ms`;
    }
  });

  // 3. Save User Preferences
  toggleAutoScan.addEventListener("change", () => {
    const isEnabled = toggleAutoScan.checked;
    chrome.storage.local.set({ autoScanEnabled: isEnabled });
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach((tab) => {
        if (tab.id) {
          chrome.tabs.sendMessage(tab.id, { action: "TOGGLE_SCAN", enabled: isEnabled }).catch(() => {});
        }
      });
    });
  });

  dwellSlider.addEventListener("input", () => {
    dwellVal.textContent = `${dwellSlider.value}ms`;
    chrome.storage.local.set({ dwellTimeMs: parseInt(dwellSlider.value, 10) });
  });

  // 4. Manual Verification
  checkBtn.addEventListener("click", () => {
    const url = manualUrlInput.value.trim();
    if (!url) return;

    checkBtn.textContent = "Checking...";
    checkBtn.disabled = true;
    resultBox.classList.add("hidden");

    chrome.runtime.sendMessage({ action: "CHECK_IMAGE", imageUrl: url }, (response) => {
      checkBtn.textContent = "Verify";
      checkBtn.disabled = false;

      if (response && response.success && response.data) {
        const d = response.data;
        resultBox.classList.remove("hidden");
        resLabel.textContent = d.label;
        resConf.textContent = `${d.confidence_pct}%`;
        
        if (d.is_ai) {
          resLabel.style.color = "#ef4444";
        } else {
          resLabel.style.color = "#10b981";
        }

        resSpatial.textContent = `${Math.round((d.spatial_ai_probability || 0) * 100)}% AI`;
        resSpectral.textContent = `${Math.round((d.spectral_ai_probability || 0) * 100)}% AI`;
      } else {
        const errorMsg = response?.error || "Unknown error";
        if (url.startsWith("blob:")) {
          alert("Cannot verify 'blob:' URLs directly because they are temporary internal browser memory objects (from ChatGPT/web apps).\n\nTo verify this image:\n1. Right-click the image on the web page and select 'Verify image with VeriSight'\nOR\n2. Use a standard web URL (http:// or https://).");
        } else {
          alert(`Verification failed: ${errorMsg}\n\nIf the server stopped, check terminal (python src/inference/app.py).`);
        }
      }
    });
  });
});
