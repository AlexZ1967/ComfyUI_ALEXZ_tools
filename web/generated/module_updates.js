// frontend/src/module_updates.ts
var labels = {
  safe: "\u{1F7E2} \u041A\u043E\u043D\u0444\u043B\u0438\u043A\u0442\u043E\u0432 \u043D\u0435 \u043E\u0431\u043D\u0430\u0440\u0443\u0436\u0435\u043D\u043E",
  risk: "\u{1F534} \u0422\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044F \u0438\u0437\u043C\u0435\u043D\u0435\u043D\u0438\u0435 \u043E\u043A\u0440\u0443\u0436\u0435\u043D\u0438\u044F \u0438\u043B\u0438 \u043E\u0442\u0434\u0435\u043B\u044C\u043D\u0430\u044F \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0430",
  unknown: "\u{1F7E1} \u041F\u0440\u043E\u0432\u0435\u0440\u043A\u0430 \u043D\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043D\u0430",
  blocked: "\u26D4 Git-\u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u0437\u0430\u0431\u043B\u043E\u043A\u0438\u0440\u043E\u0432\u0430\u043D\u043E",
  up_to_date: "\u2713 \u0423\u0436\u0435 \u0430\u043A\u0442\u0443\u0430\u043B\u0435\u043D"
};
function mountModuleUpdates(root, options) {
  const document = root.ownerDocument;
  const host = document.createElement("section");
  host.className = "alexz-update-planner";
  host.setAttribute("aria-label", "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F \u043C\u043E\u0434\u0443\u043B\u0435\u0439 \u0441 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u043E\u0439 \u0437\u0430\u0432\u0438\u0441\u0438\u043C\u043E\u0441\u0442\u0435\u0439");
  host.hidden = true;
  const title = document.createElement("strong");
  title.textContent = "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F \u043C\u043E\u0434\u0443\u043B\u0435\u0439";
  host.append(title);
  const actions = document.createElement("div");
  actions.className = "alexz-update-planner-actions";
  const summary = options.alertText;
  options.alert.classList.add("alexz-update-summary");
  summary.classList.add("alexz-update-planner-summary");
  summary.setAttribute("role", "status");
  const detailedSummary = document.createElement("div");
  detailedSummary.className = "alexz-update-planner-diagnostics";
  const results = document.createElement("div");
  host.append(actions, detailedSummary, results);
  options.alert.after(host);
  let disposed = false;
  let pending = false;
  let queued = false;
  let state = null;
  let expanded = false;
  let refreshStarted = false;
  let completedPlanId = "";
  let timer = null;
  const controller = new AbortController();
  const confirm = options.confirm ?? ((message) => window.confirm(message));
  function button(parent, text, action) {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "alexz-mod-picker-btn-small";
    node.textContent = text;
    node.onclick = async () => {
      if (pending || disposed || queued && node.dataset.cancel !== "1" && node.dataset.details !== "1") return;
      pending = true;
      syncDisabled();
      try {
        await action();
      } catch (error) {
        if (!disposed) summary.textContent = error instanceof Error ? error.message : String(error);
      } finally {
        pending = false;
        if (!disposed) syncDisabled();
      }
    };
    parent.append(node);
    return node;
  }
  function syncDisabled() {
    for (const node of host.querySelectorAll("button")) {
      node.disabled = pending || queued && node.dataset.cancel !== "1" || state?.phase === "checking" || node.dataset.apply === "1" && (!state?.plan || Date.now() / 1e3 > state.plan.expires_at);
    }
    options.refreshButton.disabled = pending || queued || state?.phase === "checking";
    detailsButton.disabled = pending || state?.phase === "checking";
  }
  async function request(route, payload) {
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(15e3)]);
    const response = await options.fetchApi(`/alexz_tools/update_plan_${route}`, {
      signal,
      cache: "no-store",
      ...payload ? {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Alexz-Update": "1" },
        body: JSON.stringify(payload)
      } : {}
    });
    if (response.status === 404) throw new Error("\u041D\u043E\u0432\u044B\u0435 backend routes \u0435\u0449\u0451 \u043D\u0435 \u0437\u0430\u0433\u0440\u0443\u0436\u0435\u043D\u044B. \u041F\u0435\u0440\u0435\u0437\u0430\u043F\u0443\u0441\u0442\u0438\u0442\u0435 ComfyUI.");
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(String(data.error ?? `HTTP ${response.status}`));
    return data;
  }
  async function poll() {
    if (timer) clearTimeout(timer);
    const data = await request("status");
    if (disposed) return;
    state = data;
    if (refreshStarted && state.phase === "ready" && state.plan && completedPlanId !== state.plan.id) {
      completedPlanId = state.plan.id;
      pending = true;
      syncDisabled();
      try {
        await options.onPlanReady();
      } finally {
        pending = false;
      }
      if (disposed) return;
    }
    render();
    if (state.phase === "checking" || state.execution?.phase === "waiting_for_shutdown") {
      timer = setTimeout(() => {
        poll().catch((error) => {
          if (!disposed) summary.textContent = error.message;
        });
      }, 2e3);
    }
  }
  async function check() {
    if (timer) clearTimeout(timer);
    expanded = false;
    host.hidden = true;
    detailsButton.setAttribute("aria-expanded", "false");
    options.alert.style.display = "block";
    summary.textContent = "\u041F\u0440\u043E\u0432\u0435\u0440\u043A\u0430 upstream \u0438 \u0437\u0430\u0432\u0438\u0441\u0438\u043C\u043E\u0441\u0442\u0435\u0439\u2026";
    refreshStarted = true;
    await request("check", {});
    await poll();
  }
  async function apply(module, mode) {
    const plan = state?.plan;
    if (!plan || Date.now() / 1e3 > plan.expires_at) throw new Error("\u041F\u043B\u0430\u043D \u0443\u0441\u0442\u0430\u0440\u0435\u043B. \u041F\u043E\u0432\u0442\u043E\u0440\u0438\u0442\u0435 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0443.");
    const selection = module ? module.module : `${plan.batch_count} \u043C\u043E\u0434\u0443\u043B\u0435\u0439`;
    const changes = (module ? module.additions : plan.batch.additions).map((item) => `${item.name}==${item.version}`).join(", ");
    const risk = mode === "code_only" ? "\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u0441\u044F \u0442\u043E\u043B\u044C\u043A\u043E \u043A\u043E\u0434. \u0417\u0430\u0432\u0438\u0441\u0438\u043C\u043E\u0441\u0442\u0438 \u0441\u043E\u0445\u0440\u0430\u043D\u044F\u0442\u0441\u044F; \u043C\u043E\u0434\u0443\u043B\u044C \u043C\u043E\u0436\u0435\u0442 \u043F\u0435\u0440\u0435\u0441\u0442\u0430\u0442\u044C \u0437\u0430\u0433\u0440\u0443\u0436\u0430\u0442\u044C\u0441\u044F.\n" : "";
    const message = `${risk}\u041F\u043E\u0441\u0442\u0430\u0432\u0438\u0442\u044C \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 ${selection} \u0432 \u043E\u0447\u0435\u0440\u0435\u0434\u044C?
` + (mode === "checked" ? `\u0414\u043E\u0431\u0430\u0432\u043B\u0435\u043D\u0438\u0435 \u043F\u0430\u043A\u0435\u0442\u043E\u0432: ${changes || "\u043D\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044F"}.
` : "") + "\u0417\u0430\u0442\u0435\u043C \u043E\u0441\u0442\u0430\u043D\u043E\u0432\u0438\u0442\u0435 ComfyUI \u0438 \u0434\u043E\u0436\u0434\u0438\u0442\u0435\u0441\u044C phase=done \u0432 status.json \u043F\u0435\u0440\u0435\u0434 \u0437\u0430\u043F\u0443\u0441\u043A\u043E\u043C. " + (mode === "checked" && changes ? "\u0414\u043E \u0443\u0441\u0442\u0430\u043D\u043E\u0432\u043A\u0438 \u043F\u0430\u043A\u0435\u0442\u043E\u0432 \u0431\u0443\u0434\u0435\u0442 \u0441\u043E\u0437\u0434\u0430\u043D\u0430 \u0440\u0435\u0437\u0435\u0440\u0432\u043D\u0430\u044F \u043A\u043E\u043F\u0438\u044F \u043E\u043A\u0440\u0443\u0436\u0435\u043D\u0438\u044F." : "");
    if (!confirm(message)) return;
    const response = await request("apply", {
      plan_id: plan.id,
      module: module?.module,
      mode,
      confirmed: true,
      acknowledge_risk: mode === "code_only"
    });
    queued = true;
    summary.textContent = `${String(response.message)}
\u0414\u0438\u0430\u0433\u043D\u043E\u0441\u0442\u0438\u043A\u0430: ${String(response.directory)}`;
    syncDisabled();
    await poll();
  }
  function render() {
    if (!state) return;
    options.alert.style.display = "block";
    summary.textContent = state.message;
    detailedSummary.textContent = state.message;
    results.replaceChildren();
    const plan = state.plan;
    if (plan) {
      const changed = plan.results.filter((item) => item.before && item.target && item.before !== item.target).length;
      summary.textContent = `\u041D\u0430\u0439\u0434\u0435\u043D\u043E \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0439: ${changed} \u043C\u043E\u0434\u0443\u043B\u0435\u0439.`;
      detailedSummary.textContent += `
\u041F\u0440\u043E\u0432\u0435\u0440\u0435\u043D\u043D\u044B\u0445 \u0434\u043B\u044F \u043E\u0431\u0449\u0435\u0433\u043E \u0430\u043F\u0434\u0435\u0439\u0442\u0430: ${plan.batch_count}. \u0421 \u0440\u0438\u0441\u043A\u043E\u043C: ${plan.counts.risk}. \u041D\u0435 \u043F\u0440\u043E\u0432\u0435\u0440\u0435\u043D\u043E: ${plan.counts.unknown}. \u0417\u0430\u0431\u043B\u043E\u043A\u0438\u0440\u043E\u0432\u0430\u043D\u043E: ${plan.counts.blocked}. \u0410\u043A\u0442\u0443\u0430\u043B\u044C\u043D\u043E: ${plan.counts.up_to_date}.`;
      if (plan.batch_error) detailedSummary.textContent += `
\u041E\u0431\u0449\u0438\u0439 \u043D\u0430\u0431\u043E\u0440 \u043D\u0435\u0441\u043E\u0432\u043C\u0435\u0441\u0442\u0438\u043C: ${plan.batch_error}`;
      if (plan.batch_count > 0) {
        const all = button(results, `\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u044C \u043F\u0440\u043E\u0432\u0435\u0440\u0435\u043D\u043D\u044B\u0435 (${plan.batch_count})`, () => apply(null, "checked"));
        all.dataset.apply = "1";
      }
      if (plan.baseline.length) {
        const old = document.createElement("details");
        const label = document.createElement("summary");
        label.textContent = `\u0418\u0441\u0445\u043E\u0434\u043D\u044B\u0435 \u043F\u0440\u043E\u0431\u043B\u0435\u043C\u044B \u043E\u043A\u0440\u0443\u0436\u0435\u043D\u0438\u044F (${plan.baseline.length})`;
        const text = document.createElement("pre");
        text.textContent = plan.baseline.join("\n");
        old.append(label, text);
        results.append(old);
      }
      for (const item of plan.results) {
        const card = document.createElement("details");
        card.dataset.module = item.module;
        card.dataset.status = item.status;
        const label = document.createElement("summary");
        label.textContent = `${item.module}: ${labels[item.status]}`;
        const reason = document.createElement("div");
        reason.textContent = item.reasons.join("\n");
        card.append(label, reason);
        if (item.target) {
          const commits = document.createElement("div");
          commits.textContent = `${item.before?.slice(0, 10)} \u2192 ${item.target.slice(0, 10)}`;
          card.append(commits);
        }
        if (item.requirements_changed !== null) {
          const dependency = document.createElement("div");
          dependency.textContent = item.requirements_changed ? "Requirements \u0438\u0437\u043C\u0435\u043D\u0438\u043B\u0438\u0441\u044C" : "Requirements \u043D\u0435 \u0438\u0437\u043C\u0435\u043D\u0438\u043B\u0438\u0441\u044C";
          card.append(dependency);
        }
        if (item.diff) {
          const diff = document.createElement("pre");
          diff.textContent = item.diff;
          card.append(diff);
        }
        if (item.additions.length) {
          const packages = document.createElement("div");
          packages.textContent = `\u0414\u043E\u0431\u0430\u0432\u044F\u0442\u0441\u044F: ${item.additions.map((pkg) => `${pkg.name}==${pkg.version}`).join(", ")}`;
          card.append(packages);
        }
        if (item.status === "safe") {
          const checked = button(card, "\u0410\u043F\u0434\u0435\u0439\u0442", () => apply(item, "checked"));
          checked.dataset.apply = "1";
        }
        if ((item.status === "risk" || item.status === "unknown") && item.before && item.target) {
          const risky = button(card, "\u0410\u043F\u0434\u0435\u0439\u0442 \u0442\u043E\u043B\u044C\u043A\u043E \u043A\u043E\u0434\u0430\u2026", () => apply(item, "code_only"));
          risky.dataset.apply = "1";
        }
        results.append(card);
      }
    }
    if (state.execution) {
      summary.textContent = state.execution.phase === "waiting_for_shutdown" ? "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043E\u0436\u0438\u0434\u0430\u0435\u0442 \u043E\u0441\u0442\u0430\u043D\u043E\u0432\u043A\u0438 ComfyUI." : `\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435: ${state.execution.phase}.`;
      detailedSummary.textContent += `
${state.execution.phase}: ${state.execution.message}
${state.execution.directory}`;
      queued = !["done", "error", "cancelled"].includes(state.execution.phase);
      if (state.execution.phase === "waiting_for_shutdown" && plan) {
        const cancel = button(results, "\u041E\u0442\u043C\u0435\u043D\u0438\u0442\u044C \u043E\u0436\u0438\u0434\u0430\u044E\u0449\u0435\u0435 \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435", async () => {
          await request("cancel", { plan_id: plan.id });
          await poll();
        });
        cancel.dataset.cancel = "1";
      }
    }
    syncDisabled();
  }
  const detailsButton = button(options.alert, "\u041F\u043E\u0434\u0440\u043E\u0431\u043D\u0435\u0435", async () => {
    if (!state) await poll();
    expanded = !expanded;
    host.hidden = !expanded;
    detailsButton.setAttribute("aria-expanded", String(expanded));
  });
  detailsButton.dataset.details = "1";
  detailsButton.classList.add("alexz-update-details-button");
  detailsButton.setAttribute("aria-expanded", "false");
  button(actions, "\u0421\u0442\u0430\u0442\u0443\u0441 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0438 / \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F", poll);
  const refresh = async () => {
    if (pending || queued || disposed || state?.phase === "checking") return;
    pending = true;
    syncDisabled();
    try {
      await check();
    } catch (error) {
      if (!disposed) summary.textContent = error instanceof Error ? error.message : String(error);
    } finally {
      pending = false;
      if (!disposed) syncDisabled();
    }
  };
  const dispose = () => {
    disposed = true;
    controller.abort();
    if (timer) clearTimeout(timer);
    detailsButton.remove();
    host.remove();
  };
  return { refresh, dispose };
}
export {
  mountModuleUpdates
};
//# sourceMappingURL=module_updates.js.map
