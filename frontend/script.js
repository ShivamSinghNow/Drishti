(() => {
  const root = document.documentElement;
  const storedTheme = window.localStorage.getItem("drishti-theme");
  const API_URL = window.DRISHTI_API?.analyzeUrl || "/api/analyze";
  const MODEL = window.DRISHTI_MODEL || {};
  const SAMPLE_IMAGE = "assets/xray-sample.png";

  const formatPercent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
  const formatMetric = (value, digits = 3) => Number(value).toFixed(digits);

  const setText = (selector, value) => {
    if (value == null || value === "") return;
    document.querySelectorAll(selector).forEach((node) => {
      node.textContent = value;
    });
  };

  const initModelInfo = () => {
    const { validation = {}, gguf = {}, hosting = {}, site = {} } = MODEL;
    if (validation.accuracy != null) {
      setText("[data-model-accuracy]", formatPercent(validation.accuracy));
      setText("[data-model-hero-accuracy]", formatPercent(validation.accuracy));
    }
    if (validation.macroF1 != null) setText("[data-model-macro-f1]", formatMetric(validation.macroF1));
    if (validation.activeTbF1 != null) setText("[data-model-active-f1]", formatMetric(validation.activeTbF1));
    if (validation.samples != null) {
      setText("[data-model-val-split]", `Run #4 · ${validation.samples.toLocaleString()} val samples`);
    }

    setText("[data-model-hero-live]", `GPU-hosted · run4 LoRA`);
    setText("[data-model-hero-gguf]", `${gguf.quant || "Q4_K_M"} · ${gguf.size || "4.68 GB"}`);
    setText("[data-model-gguf-quant]", gguf.quant);
    setText("[data-model-gguf-size-p]", `${gguf.size || "4.68 GB"} on Hugging Face`);
    setText(
      "[data-model-checkpoint]",
      `step ${MODEL.checkpointStep || 4950} · ${MODEL.baseModel || "Qwen2-VL-7B-Instruct"}`,
    );
    setText(
      "[data-model-gguf-files]",
      `${gguf.quant || "Q4_K_M"} text (${gguf.size || "4.68 GB"}) + mmproj (${gguf.mmprojSize || "1.26 GiB"})`,
    );
    setText("[data-model-modal-host]", `GPU-hosted · run4 LoRA`);
    setText("[data-model-modal-gpu]", "GPU-hosted");
    setText("[data-model-modal-quant]", hosting.quant || "4-bit NF4 LoRA");
    setText("[data-model-cold-start]", hosting.coldStartNote || "");

    const curl = document.querySelector("[data-terminal-curl]");
    if (curl) {
      curl.textContent = '$ curl -F "image=@scan.png" /api/analyze';
    }

    const terminalModel = document.querySelector("[data-terminal-model]");
    if (terminalModel && !terminalModel.dataset.dynamic) {
      terminalModel.textContent = `run4 · step ${MODEL.checkpointStep || 4950}`;
    }

    if (site.hfGguf) {
      document.querySelectorAll("[data-model-gguf-link]").forEach((link) => {
        link.href = site.hfGguf;
      });
    }
  };

  initModelInfo();

  const LABEL_CLASS = {
    active_tb: "label-tb",
    healthy: "label-healthy",
    sick_but_non_tb: "label-other",
  };

  if (storedTheme === "dark" || storedTheme === "light") {
    root.dataset.theme = storedTheme;
  }

  document.querySelector("[data-theme-toggle]")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    window.localStorage.setItem("drishti-theme", next);
  });

  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (event) => {
      const id = anchor.getAttribute("href");
      if (!id || id === "#") return;

      const target = id === "#top" ? document.body : document.querySelector(id);
      if (!target) return;

      event.preventDefault();
      const navHeight = document.querySelector("[data-nav]")?.offsetHeight ?? 70;
      const top = target === document.body
        ? 0
        : target.getBoundingClientRect().top + window.scrollY - navHeight - 10;

      window.scrollTo({ top: Math.max(top, 0), behavior: reducedMotion() ? "auto" : "smooth" });
      window.history.replaceState(null, "", id);
    });
  });

  const reveals = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });

    reveals.forEach((el) => observer.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("is-visible"));
  }

  const demo = document.querySelector("[data-demo]");
  if (demo) {
    const logOutput = demo.querySelector("[data-log-output]");
    const jsonOutput = demo.querySelector("[data-json-output]");
    const stateText = demo.querySelector("[data-run-state]");
    const percentText = demo.querySelector("[data-phone-percent]");
    const phoneHead = demo.querySelector("[data-phone-head]");
    const progressBar = demo.querySelector("[data-phone-progress]");
    const runButtons = document.querySelectorAll("[data-run-demo]");
    const resetButtons = document.querySelectorAll("[data-reset-demo]");
    const uploadZone = demo.querySelector("[data-upload-zone]");
    const fileInput = demo.querySelector("[data-file-input]");
    const scanImage = demo.querySelector("[data-scan-image]");
    const resultImage = demo.querySelector("[data-result-image]");
    const resultLabel = demo.querySelector("[data-result-label]");
    const resultConfidence = demo.querySelector("[data-result-confidence]");
    const resultGuidance = demo.querySelector("[data-result-guidance]");
    const resultLatency = demo.querySelectorAll("[data-result-latency]");
    const resultProbabilities = demo.querySelector("[data-result-probabilities]");
    const errorBanner = demo.querySelector("[data-demo-error]");
    const terminalModel = demo.querySelector("[data-terminal-model]");

    let intervalId = 0;
    let timeouts = [];
    let selectedFile = null;
    let previewUrl = null;
    let activeRequest = null;

    const clearTimers = () => {
      window.clearInterval(intervalId);
      intervalId = 0;
      timeouts.forEach((id) => window.clearTimeout(id));
      timeouts = [];
    };

    const setProgress = (value) => {
      const safeValue = Math.max(0, Math.min(100, Math.round(value)));
      demo.style.setProperty("--demo-progress", `${safeValue}%`);
      demo.style.setProperty("--scan-progress", `${safeValue}%`);
      if (percentText) percentText.textContent = String(safeValue);
      if (progressBar) progressBar.style.width = `${safeValue}%`;
      if (phoneHead) phoneHead.textContent = safeValue < 100 ? "running" : "done";
    };

    const setStage = (stage) => {
      demo.dataset.stage = stage;
      if (stateText) {
        stateText.textContent = stage === "idle" ? "ready" : stage === "analyzing" ? "running" : stage === "error" ? "error" : "complete";
      }
    };

    const setError = (message) => {
      if (!errorBanner) return;
      if (message) {
        errorBanner.hidden = false;
        errorBanner.textContent = message;
      } else {
        errorBanner.hidden = true;
        errorBanner.textContent = "";
      }
    };

    const revokePreview = () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
      }
    };

    const setPreviewImage = (source) => {
      if (scanImage) scanImage.src = source;
      if (resultImage) resultImage.src = source;
    };

    const resetResults = () => {
      if (resultLabel) {
        resultLabel.textContent = "—";
        resultLabel.className = "result-label";
      }
      if (resultConfidence) resultConfidence.textContent = "—";
      if (resultGuidance) resultGuidance.textContent = "Upload a chest X-ray to run the live Drishti run4 classifier.";
      if (resultProbabilities) resultProbabilities.innerHTML = "";
      resultLatency.forEach((node) => {
        node.textContent = "—";
      });
      if (jsonOutput) {
        jsonOutput.textContent = "{\n  \"status\": \"ready\"\n}";
      }
    };

    const reset = () => {
      if (activeRequest) {
        activeRequest.abort();
        activeRequest = null;
      }
      clearTimers();
      setStage("idle");
      setProgress(0);
      setError("");
      selectedFile = null;
      if (fileInput) fileInput.value = "";
      revokePreview();
      setPreviewImage(SAMPLE_IMAGE);
      resetResults();
      if (logOutput) logOutput.innerHTML = "";
    };

    const appendLog = ({ msg, tag }) => {
      if (!logOutput) return;
      const line = document.createElement("div");
      line.className = `log-line ${tag}`;
      line.textContent = msg;
      logOutput.appendChild(line);
      logOutput.scrollTop = logOutput.scrollHeight;
    };

    const formatPercentLocal = (value) => `${(value * 100).toFixed(1)}%`;

    const renderProbabilities = (probabilities) => {
      if (!resultProbabilities || !probabilities) return;
      resultProbabilities.innerHTML = "";
      Object.entries(probabilities).forEach(([label, score]) => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `<span>${label}</span><strong>${formatPercentLocal(score)}</strong>`;
        resultProbabilities.appendChild(row);
      });
    };

    const renderResult = (payload) => {
      const label = payload.label || "unknown";
      const display = payload.label_display || label;
      const confidence = payload.confidence ?? 0;
      const latency = payload.latency_ms ?? 0;

      if (resultLabel) {
        resultLabel.textContent = display;
        resultLabel.className = `result-label ${LABEL_CLASS[label] || ""}`.trim();
      }
      if (resultConfidence) {
        resultConfidence.innerHTML = `<b>${formatPercentLocal(confidence)}</b> confidence`;
      }
      if (resultGuidance) {
        resultGuidance.textContent = payload.guidance || payload.disclaimer || "";
      }
      resultLatency.forEach((node) => {
        node.textContent = `${(latency / 1000).toFixed(2)}s`;
      });
      renderProbabilities(payload.class_probabilities);
      if (jsonOutput) {
        jsonOutput.textContent = JSON.stringify(payload, null, 2);
      }
      if (terminalModel && payload.model) {
        terminalModel.dataset.dynamic = "true";
        terminalModel.textContent = payload.model.checkpoint || payload.model.adapter || payload.model.gguf_artifact;
      }
    };

    const animateProgress = (duration) => {
      const start = Date.now();
      intervalId = window.setInterval(() => {
        const elapsed = Date.now() - start;
        const progress = Math.min(92, (elapsed / duration) * 92);
        setProgress(progress);
      }, 40);
    };

    async function blobFromSource(useSample) {
      if (!useSample && selectedFile) {
        return selectedFile;
      }
      const response = await fetch(SAMPLE_IMAGE);
      if (!response.ok) {
        throw new Error("Could not load the bundled sample X-ray.");
      }
      const blob = await response.blob();
      return new File([blob], "sample-xray.png", { type: blob.type || "image/png" });
    }

    async function run({ useSample = false } = {}) {
      if (activeRequest) {
        activeRequest.abort();
      }

      clearTimers();
      setStage("analyzing");
      setProgress(0);
      setError("");
      if (logOutput) logOutput.innerHTML = "";

      appendLog({ msg: "-> POST /api/analyze", tag: "sys" });
      appendLog({ msg: `-> load ${MODEL.name || "run4 vision-lora"}`, tag: "sys" });
      appendLog({ msg: "-> preprocess: resize 512x512, normalize", tag: "inf" });

      const duration = reducedMotion() ? 800 : 4500;
      animateProgress(duration);

      const controller = new AbortController();
      activeRequest = controller;

      try {
        const file = await blobFromSource(useSample);
        revokePreview();
        previewUrl = URL.createObjectURL(file);
        setPreviewImage(previewUrl);

        const formData = new FormData();
        formData.append("image", file, file.name);

        appendLog({ msg: "-> vision encoder + language head scoring", tag: "inf" });

        const response = await fetch(API_URL, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || payload.error || `Inference failed (${response.status}).`);
        }

        clearTimers();
        setProgress(100);
        appendLog({
          msg: `[ok] ${payload.label_display || payload.label} (${formatPercentLocal(payload.confidence || 0)})`,
          tag: "ok",
        });
        renderResult(payload);
        timeouts.push(window.setTimeout(() => setStage("result"), reducedMotion() ? 50 : 180));
      } catch (error) {
        clearTimers();
        setProgress(0);
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        const message = error instanceof Error ? error.message : "Inference request failed.";
        appendLog({ msg: `[error] ${message}`, tag: "err" });
        setError(message);
        setStage("error");
      } finally {
        activeRequest = null;
      }
    }

    runButtons.forEach((button) => {
      button.addEventListener("click", () => {
        const useSample = button.hasAttribute("data-use-sample");
        run({ useSample });
      });
    });

    resetButtons.forEach((button) => button.addEventListener("click", reset));

    uploadZone?.addEventListener("click", () => fileInput?.click());

    fileInput?.addEventListener("change", () => {
      const [file] = fileInput.files || [];
      if (!file) return;
      selectedFile = file;
      revokePreview();
      previewUrl = URL.createObjectURL(file);
      setPreviewImage(previewUrl);
      appendLog({ msg: `-> selected ${file.name}`, tag: "sys" });
      run({ useSample: false });
    });

    reset();
  }

  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      const id = button.dataset.tab;
      const tabList = button.closest(".tabs");
      const panelRoot = tabList?.parentElement;
      if (!id || !tabList || !panelRoot) return;

      tabList.querySelectorAll("[data-tab]").forEach((tab) => {
        const active = tab === button;
        tab.classList.toggle("active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
      });

      panelRoot.querySelectorAll("[data-tab-panel]").forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.tabPanel === id);
      });
    });
  });
})();
