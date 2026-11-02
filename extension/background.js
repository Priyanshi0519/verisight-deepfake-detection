/**
 * VeriSight - Background Service Worker (Manifest V3)
 * Manages context menus and bridges network requests to the local inference server.
 */

const BACKEND_URL = "http://127.0.0.1:5000";

// 1. Setup Context Menu on Install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verisight-check-image",
    title: "Verify image with VeriSight (AI Check)",
    contexts: ["image"]
  });
  console.log("VeriSight extension initialized with context menu.");
});

// 2. Handle Context Menu Clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "verisight-check-image" && info.srcUrl && tab?.id) {
    chrome.tabs.sendMessage(tab.id, {
      action: "VERIFY_SINGLE_IMAGE",
      srcUrl: info.srcUrl
    });
  }
});

// 3. Message Listener for Content Scripts and Popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "HEALTH_CHECK") {
    fetch(`${BACKEND_URL}/health`, { method: "GET" })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }

  if (request.action === "CHECK_IMAGE") {
    const payload = request.imageBase64
      ? { image_base64: request.imageBase64 }
      : { image_url: request.imageUrl };

    fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(async res => {
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Inference error");
        sendResponse({ success: true, data: json });
      })
      .catch(err => {
        sendResponse({ success: false, error: err.message });
      });
    return true;
  }

  if (request.action === "CHECK_BATCH") {
    fetch(`${BACKEND_URL}/predict-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls: request.urls })
    })
      .then(async res => {
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Batch inference error");
        sendResponse({ success: true, data: json });
      })
      .catch(err => {
        sendResponse({ success: false, error: err.message });
      });
    return true;
  }
});
