/* Interactive teaching widgets. Vanilla JS, no deps.
   Each init is idempotent and re-runs on Material's instant navigation. */

(function () {
  "use strict";

  const BRAND = "#6c63ff";

  /* ---- 1. Bootstrap CI visualizer (stats-explained) ---------------------- */

  function initBootstrap() {
    const root = document.getElementById("ix-bootstrap");
    if (!root || root.dataset.init) return;
    root.dataset.init = "1";

    const SEEDS = [0.9201, 0.9265, 0.9189, 0.9249, 0.9233, 0.9272, 0.918, 0.9286];
    const LO = 0.915, HI = 0.933, BINS = 44;
    let means = [];

    root.innerHTML = `
      <div class="ix-head">Try it: bootstrap the CI from 8 seed accuracies</div>
      <div class="ix-controls">
        <button class="md-button" data-n="1">resample once</button>
        <button class="md-button" data-n="1000">×1000</button>
        <button class="md-button" data-n="0">reset</button>
        <span class="ix-count">0 resamples</span>
      </div>
      <div class="ix-pick">&nbsp;</div>
      <canvas width="640" height="180"></canvas>
      <div class="ix-ci">&nbsp;</div>`;

    const canvas = root.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const pickEl = root.querySelector(".ix-pick");
    const ciEl = root.querySelector(".ix-ci");
    const countEl = root.querySelector(".ix-count");

    function resample() {
      const idx = Array.from({ length: 8 }, () => Math.floor(Math.random() * 8));
      const mean = idx.reduce((s, i) => s + SEEDS[i], 0) / 8;
      means.push(mean);
      return { idx, mean };
    }

    function percentile(sorted, p) {
      const k = (sorted.length - 1) * p;
      const f = Math.floor(k);
      return sorted[f] + (sorted[Math.min(f + 1, sorted.length - 1)] - sorted[f]) * (k - f);
    }

    function draw() {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const x = (v) => ((v - LO) / (HI - LO)) * w;

      // CI band first (behind bars)
      let lo95 = null, hi95 = null;
      if (means.length >= 50) {
        const s = [...means].sort((a, b) => a - b);
        lo95 = percentile(s, 0.025);
        hi95 = percentile(s, 0.975);
        ctx.fillStyle = "rgba(108,99,255,0.18)";
        ctx.fillRect(x(lo95), 0, x(hi95) - x(lo95), h - 22);
      }

      // histogram of resample means
      const counts = new Array(BINS).fill(0);
      for (const m of means) {
        const b = Math.min(BINS - 1, Math.max(0, Math.floor(((m - LO) / (HI - LO)) * BINS)));
        counts[b]++;
      }
      const max = Math.max(1, ...counts);
      ctx.fillStyle = BRAND;
      counts.forEach((c, b) => {
        if (!c) return;
        const bh = (c / max) * (h - 34);
        ctx.fillRect((b / BINS) * w + 1, h - 22 - bh, w / BINS - 2, bh);
      });

      // the 8 original seeds as ticks
      ctx.fillStyle = "#999";
      SEEDS.forEach((v) => ctx.fillRect(x(v) - 1, h - 18, 2, 8));
      ctx.font = "11px var(--md-code-font-family, monospace)";
      ctx.fillText("seed accuracies", 4, h - 2);

      countEl.textContent = `${means.length} resamples`;
      ciEl.innerHTML =
        lo95 !== null
          ? `95% CI of the mean: <b>[${lo95.toFixed(4)}, ${hi95.toFixed(4)}]</b> — the shaded band`
          : "CI appears after 50 resamples…";
    }

    root.querySelectorAll("button").forEach((btn) =>
      btn.addEventListener("click", () => {
        const n = +btn.dataset.n;
        if (n === 0) {
          means = [];
          pickEl.innerHTML = "&nbsp;";
        } else if (n === 1) {
          const { idx, mean } = resample();
          pickEl.innerHTML =
            "picked " + idx.map((i) => `s${i + 1}`).join(" ") + ` → mean <b>${mean.toFixed(4)}</b>`;
        } else {
          for (let i = 0; i < n; i++) resample();
          pickEl.innerHTML = `&nbsp;`;
        }
        draw();
      })
    );
    draw();
  }

  /* ---- 2. Seeds vs significance explorer (stats-explained) --------------- */

  function initSeeds() {
    const root = document.getElementById("ix-seeds");
    if (!root || root.dataset.init) return;
    root.dataset.init = "1";

    root.innerHTML = `
      <div class="ix-head">Try it: how many seeds before Wilcoxon can even speak?</div>
      <label class="ix-slider">seeds <input type="range" min="3" max="15" value="8" />
        <b class="n">8</b></label>
      <div class="ix-row">smallest possible two-sided p = 2/2<sup class="exp">8</sup> =
        <code class="minp"></code></div>
      <div class="ix-verdict"></div>`;

    const slider = root.querySelector("input");
    function update() {
      const n = +slider.value;
      const minp = 2 / Math.pow(2, n);
      root.querySelector(".n").textContent = n;
      root.querySelector(".exp").textContent = n;
      root.querySelector(".minp").textContent =
        minp >= 0.001 ? minp.toFixed(4) : minp.toExponential(1);
      const v = root.querySelector(".ix-verdict");
      if (n < 6) {
        v.innerHTML = "❌ p can never get below 0.05 — the aggregator falls back to a <b>paired t-test</b> here";
        v.className = "ix-verdict bad";
      } else {
        v.innerHTML = `✓ Wilcoxon can reach significance — at p ≤ 0.05 ${n >= 8 ? "with room to spare" : "(barely)"}`;
        v.className = "ix-verdict good";
      }
    }
    slider.addEventListener("input", update);
    update();
  }

  /* ---- 3. Command builder (configs) -------------------------------------- */

  function initBuilder() {
    const root = document.getElementById("ix-builder");
    if (!root || root.dataset.init) return;
    root.dataset.init = "1";

    root.innerHTML = `
      <div class="ix-head">Try it: compose a run (colors = override level)</div>
      <div class="ix-grid">
        <label>experiment <select data-k="exp"><option value="">—</option>
          <option>example</option></select></label>
        <label>loss <select data-k="loss"><option value="">supervised (default)</option>
          <option value="contrastive">contrastive</option></select></label>
        <label>logger <select data-k="log"><option value="">project default</option>
          <option>wandb</option><option>trackio</option><option>csv</option></select></label>
        <label>lr <input data-k="lr" placeholder="e.g. 1e-3" size="8" /></label>
        <label>seed <input data-k="seed" placeholder="42" size="6" /></label>
      </div>
      <pre class="ix-cmd"><code></code></pre>
      <button class="md-button ix-copy">copy</button>`;

    const out = root.querySelector("code");
    function update() {
      const v = (k) => root.querySelector(`[data-k="${k}"]`).value.trim();
      let html = `<span class="tok-base">uv run python src/&lt;pkg&gt;/train.py</span>`;
      let txt = "uv run python src/<pkg>/train.py";
      if (v("exp")) { html += ` <span class="tok-preset">experiment=${v("exp")}</span>`; txt += ` experiment=${v("exp")}`; }
      if (v("loss")) { html += ` <span class="tok-group">loss=${v("loss")}</span>`; txt += ` loss=${v("loss")}`; }
      if (v("lr")) { html += ` <span class="tok-key">model.lr=${v("lr")}</span>`; txt += ` model.lr=${v("lr")}`; }
      if (v("seed")) { html += ` <span class="tok-key">seed=${v("seed")}</span>`; txt += ` seed=${v("seed")}`; }
      if (v("log")) { html += ` <span class="tok-key">logger.kind=${v("log")}</span>`; txt += ` logger.kind=${v("log")}`; }
      out.innerHTML = html;
      out.dataset.txt = txt;
    }
    root.querySelectorAll("select,input").forEach((el) => el.addEventListener("input", update));
    root.querySelector(".ix-copy").addEventListener("click", () => {
      navigator.clipboard.writeText(out.dataset.txt);
      root.querySelector(".ix-copy").textContent = "copied!";
      setTimeout(() => (root.querySelector(".ix-copy").textContent = "copy"), 1200);
    });
    update();
  }

  /* ---- 4. Generated-project explorer (getting-started) ------------------- */

  function initTree() {
    const root = document.getElementById("ix-tree");
    if (!root || root.dataset.init) return;
    root.dataset.init = "1";

    // [path, show(opts)] — the load-bearing files only
    const FILES = [
      ["src/<pkg>/configs.py", () => true],
      ["src/<pkg>/experiments.py", () => true],
      ["src/<pkg>/train.py · eval.py · training_loop.py", () => true],
      ["src/<pkg>/objectives.py", () => true],
      ["src/<pkg>/data/datamodule.py", () => true],
      ["src/<pkg>/data/openml_data.py", (o) => o.flavor === "tabular"],
      ["src/<pkg>/data/image_text.py", (o) => o.flavor === "multimodal"],
      ["src/<pkg>/models/module.py", () => true],
      ["src/<pkg>/benchmark.py · sklearn_wrapper.py", (o) => o.flavor === "tabular"],
      ["src/<pkg>/utils/cli.py · run_dir.py · stats.py · seed.py", () => true],
      ["src/<pkg>/utils/schedulers.py", (o) => o.fw === "pytorch"],
      ["src/<pkg>/utils/loggers.py", (o) => o.fw === "jax"],
      ["src/<pkg>/utils/codever.py", (o) => o.flavor === "tabular"],
      ["scripts/sweep.py · tune.py · run_seeds.sh · sbatch_*.sh", () => true],
      ["scripts/run_benchmark.sh · aggregate_benchmark.py", (o) => o.flavor === "tabular"],
      ["tests/test_cli.py", () => true],
      ["tests/test_model.py", (o) => o.example],
      ["tests/test_tabular.py · test_codever.py", (o) => o.flavor === "tabular"],
      ["demo/app.py  (HF Spaces demo)", (o) => o.example],
      ["configs/local.yaml  (gitignored machine overrides)", () => true],
    ];

    root.innerHTML = `
      <div class="ix-head">Try it: what each answer generates</div>
      <div class="ix-controls">
        <span class="ix-lbl">framework</span>
        <label><input type="radio" name="ixfw" value="pytorch" checked /> pytorch</label>
        <label><input type="radio" name="ixfw" value="jax" /> jax</label>
        <span class="ix-lbl">flavor</span>
        <label><input type="radio" name="ixfl" value="generic" checked /> generic</label>
        <label><input type="radio" name="ixfl" value="tabular" /> tabular</label>
        <label><input type="radio" name="ixfl" value="multimodal" /> multimodal</label>
        <label class="ix-ex"><input type="checkbox" checked /> example</label>
      </div>
      <pre class="ix-files"><code></code></pre>
      <div class="ix-note">&nbsp;</div>`;

    const out = root.querySelector("code");
    const note = root.querySelector(".ix-note");
    let prev = new Set();

    function update() {
      const fw = root.querySelector('[name="ixfw"]:checked').value;
      const flRadios = root.querySelectorAll('[name="ixfl"]');
      if (fw === "jax") {
        flRadios.forEach((r) => { r.disabled = r.value !== "generic"; if (r.value === "generic") r.checked = true; });
        note.textContent = "jax pairs with the generic flavor — tabular/multimodal build on PyTorch libraries";
      } else {
        flRadios.forEach((r) => (r.disabled = false));
        note.innerHTML = "&nbsp;";
      }
      const o = {
        fw,
        flavor: root.querySelector('[name="ixfl"]:checked').value,
        example: root.querySelector(".ix-ex input").checked,
      };
      const visible = FILES.filter(([, f]) => f(o)).map(([p]) => p);
      const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      out.innerHTML = visible
        .map((p) => `<span class="${prev.size && !prev.has(p) ? "ix-new" : ""}">${esc(p)}</span>`)
        .join("\n");
      prev = new Set(visible);
    }
    root.querySelectorAll("input").forEach((el) => el.addEventListener("change", update));
    update();
    prev = new Set(FILES.filter(([, f]) => f({ fw: "pytorch", flavor: "generic", example: true })).map(([p]) => p));
  }

  /* ---- boot --------------------------------------------------------------- */

  function initAll() {
    initBootstrap();
    initSeeds();
    initBuilder();
    initTree();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(initAll); // Material instant navigation
  } else {
    document.addEventListener("DOMContentLoaded", initAll);
  }
})();
