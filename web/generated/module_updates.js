// frontend/src/module_updates.ts
function mountModuleUpdates(root, options) {
  const document = root.ownerDocument;
  const host = document.createElement("section");
  host.className = "alexz-update-planner";
  host.setAttribute("aria-label", "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F \u043C\u043E\u0434\u0443\u043B\u0435\u0439 \u0441 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u043E\u0439 \u0437\u0430\u0432\u0438\u0441\u0438\u043C\u043E\u0441\u0442\u0435\u0439");
  host.hidden = true;
  const actions = document.createElement("div");
  actions.className = "alexz-update-planner-actions";
  const summary = options.alertText;
  options.alert.classList.add("alexz-update-summary");
  summary.classList.add("alexz-update-planner-summary");
  summary.setAttribute("role", "status");
  const results = document.createElement("div");
  host.append(actions, results);
  options.alert.append(host);
  let disposed = false;
  let pending = false;
  let queued = false;
  let state = null;
  let selectedCard = null;
  let selectedModule = "";
  let applying = false;
  const frozenControls = /* @__PURE__ */ new Map();
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
    node.onclick = async (event) => {
      event.stopPropagation();
      if (pending || disposed || queued && node.dataset.cancel !== "1" && node.dataset.status !== "1") return;
      pending = true;
      syncDisabled();
      try {
        await action();
      } catch (error) {
        if (!disposed) showError(error);
      } finally {
        pending = false;
        if (!disposed) syncDisabled();
      }
    };
    parent.append(node);
    return node;
  }
  function unlock() {
    root.inert = false;
    root.removeAttribute("aria-busy");
    for (const [node, disabled] of frozenControls) node.disabled = disabled;
    frozenControls.clear();
  }
  function syncDisabled() {
    unlock();
    options.refreshButton.disabled = pending || queued || state?.phase === "checking";
    updateButton.hidden = state?.phase !== "ready" || !state.plan || pending || queued;
    updateButton.textContent = `\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u044C \u043C\u043E\u0434\u0443\u043B\u0438 (${state?.plan?.batch_count ?? 0})`;
    updateButton.disabled = pending || queued || !state?.plan || state.plan.batch_count === 0 || Boolean(state.plan.batch_error) || Date.now() / 1e3 > state.plan.expires_at;
    updateButton.title = updateButton.disabled ? "\u041D\u0435\u0442 \u043F\u0440\u043E\u0432\u0435\u0440\u0435\u043D\u043D\u044B\u0445 \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0439; \u043F\u043E\u0434\u0440\u043E\u0431\u043D\u043E\u0441\u0442\u0438 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI" : "";
    const individual = selectedCard?.querySelector(".alexz-module-update-button");
    if (individual) individual.disabled = pending || queued || state?.phase !== "ready" || !state.plan || individual.dataset.checked !== "1" || Date.now() / 1e3 > state.plan.expires_at;
    if (applying || queued && state?.execution?.phase !== "waiting_for_shutdown") {
      for (const node of root.querySelectorAll("input, select, button, textarea")) {
        frozenControls.set(node, node.disabled);
        node.disabled = true;
      }
      root.inert = true;
      root.setAttribute("aria-busy", "true");
    }
  }
  function showError(error) {
    console.error("[ALEXZ_tools] \u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043C\u043E\u0434\u0443\u043B\u0435\u0439:", error);
    summary.textContent = "\u041E\u0448\u0438\u0431\u043A\u0430. \u041F\u043E\u0434\u0440\u043E\u0431\u043D\u043E\u0441\u0442\u0438 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI.";
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
    options.onModulesUpdated?.(state.updated_modules ?? []);
    render();
    if (state.phase === "checking" || state.execution && !["done", "error", "cancelled"].includes(state.execution.phase)) {
      timer = setTimeout(() => {
        poll().catch((error) => {
          if (!disposed) showError(error);
        });
      }, 2e3);
    }
  }
  async function check() {
    if (timer) clearTimeout(timer);
    options.alert.style.display = "block";
    summary.textContent = "\u041F\u0440\u043E\u0432\u0435\u0440\u043A\u0430 upstream \u0438 \u0437\u0430\u0432\u0438\u0441\u0438\u043C\u043E\u0441\u0442\u0435\u0439\u2026";
    refreshStarted = true;
    await request("check", {});
    await poll();
  }
  async function apply(module) {
    const plan = state?.plan;
    if (!plan || Date.now() / 1e3 > plan.expires_at) throw new Error("\u041F\u043B\u0430\u043D \u0443\u0441\u0442\u0430\u0440\u0435\u043B. \u041F\u043E\u0432\u0442\u043E\u0440\u0438\u0442\u0435 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0443.");
    const changes = (module?.additions ?? plan.batch.additions).map((item) => `${item.name}==${item.version}`).join(", ");
    const selection = module ? module.module : `${plan.batch_count} \u043C\u043E\u0434\u0443\u043B\u0435\u0439`;
    if (!confirm(`\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u044C ${selection}?
\u0414\u043E\u0431\u0430\u0432\u043B\u0435\u043D\u0438\u0435 \u043F\u0430\u043A\u0435\u0442\u043E\u0432: ${changes || "\u043D\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044F"}.
\u041F\u043E\u0441\u043B\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043D\u0438\u044F \u043F\u0435\u0440\u0435\u0437\u0430\u043F\u0443\u0441\u0442\u0438\u0442\u0435 ComfyUI.`)) return;
    applying = true;
    syncDisabled();
    try {
      await request("apply", { plan_id: plan.id, module: module?.module, mode: "checked", confirmed: true });
      queued = true;
      await poll();
    } finally {
      applying = false;
      syncDisabled();
    }
  }
  function render() {
    if (!state) return;
    options.alert.style.display = "block";
    results.replaceChildren();
    summary.textContent = state.phase === "checking" ? state.message : state.phase === "error" ? "\u041E\u0448\u0438\u0431\u043A\u0430 \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0438. \u041F\u043E\u0434\u0440\u043E\u0431\u043D\u043E\u0441\u0442\u0438 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI." : "\u041F\u0440\u043E\u0432\u0435\u0440\u043A\u0430 \u0435\u0449\u0451 \u043D\u0435 \u0437\u0430\u043F\u0443\u0441\u043A\u0430\u043B\u0430\u0441\u044C";
    if (state.plan) {
      const changed = state.plan.results.filter((item) => item.before && item.target && item.before !== item.target).length;
      summary.textContent = `\u041D\u0430\u0439\u0434\u0435\u043D\u043E \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0439: ${changed} \u043C\u043E\u0434\u0443\u043B\u0435\u0439.`;
    }
    queued = Boolean(state.execution && !["done", "error", "cancelled"].includes(state.execution.phase));
    if (state.execution && !(state.plan && state.execution.phase === "done")) {
      summary.textContent = state.execution.phase === "done" ? "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043D\u043E. \u041F\u0435\u0440\u0435\u0437\u0430\u043F\u0443\u0441\u0442\u0438\u0442\u0435 ComfyUI." : state.execution.phase === "error" ? "\u041E\u0448\u0438\u0431\u043A\u0430 \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u044F. \u041F\u043E\u0434\u0440\u043E\u0431\u043D\u043E\u0441\u0442\u0438 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI." : state.execution.phase === "cancelled" ? "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043E\u0442\u043C\u0435\u043D\u0435\u043D\u043E." : "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u0432\u044B\u043F\u043E\u043B\u043D\u044F\u0435\u0442\u0441\u044F. \u0425\u043E\u0434 \u0440\u0430\u0431\u043E\u0442\u044B \u2014 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI.";
      if (state.execution.phase === "waiting_for_shutdown") {
        summary.textContent = "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043E\u0436\u0438\u0434\u0430\u0435\u0442 \u043E\u0441\u0442\u0430\u043D\u043E\u0432\u043A\u0438 ComfyUI.";
        const cancel = button(results, "\u041E\u0442\u043C\u0435\u043D\u0438\u0442\u044C \u043E\u0436\u0438\u0434\u0430\u044E\u0449\u0435\u0435 \u043E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435", async () => {
          await request("cancel", { plan_id: state?.execution?.directory.split(/[\\/]/).pop() });
          await poll();
        });
        cancel.dataset.cancel = "1";
      }
    }
    host.hidden = state.execution?.phase !== "waiting_for_shutdown";
    renderSelectedCard();
    syncDisabled();
  }
  function renderSelectedCard() {
    selectedCard?.querySelector(".alexz-module-update-button")?.remove();
    if (!selectedCard?.isConnected) return;
    const status = selectedCard.querySelector(".alexz-module-runtime-status");
    if (status) {
      if (!status.dataset.originalHtml) {
        status.dataset.originalHtml = status.innerHTML;
        status.dataset.originalClass = status.className;
      }
      if (state?.restart_required_modules?.includes(selectedModule)) {
        status.classList.remove("ok", "unknown");
        status.classList.add("warn");
        const value = status.querySelector("span:last-child");
        if (value) value.textContent = "\u0422\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044F \u043F\u0435\u0440\u0435\u0437\u0430\u0433\u0440\u0443\u0437\u043A\u0430";
      } else {
        status.innerHTML = status.dataset.originalHtml;
        status.className = status.dataset.originalClass ?? status.className;
      }
    }
    const item = state?.plan?.results.find((item2) => item2.module === selectedModule);
    if (!item?.before || !item.target || item.before === item.target) return;
    const action = button(selectedCard, "\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u044C \u043C\u043E\u0434\u0443\u043B\u044C", () => apply(item));
    action.classList.add("alexz-module-update-button");
    action.dataset.checked = item.status === "safe" ? "1" : "0";
    if (item.status !== "safe") action.title = "\u041E\u0431\u043D\u043E\u0432\u043B\u0435\u043D\u0438\u0435 \u043D\u0435 \u043F\u0440\u043E\u0448\u043B\u043E \u043F\u0440\u043E\u0432\u0435\u0440\u043A\u0443; \u043F\u043E\u0434\u0440\u043E\u0431\u043D\u043E\u0441\u0442\u0438 \u0432 \u043A\u043E\u043D\u0441\u043E\u043B\u0438 ComfyUI";
  }
  const updateButton = button(options.alert, "\u041E\u0431\u043D\u043E\u0432\u0438\u0442\u044C \u043C\u043E\u0434\u0443\u043B\u0438 (0)", () => apply());
  options.alert.style.display = "block";
  syncDisabled();
  const renderModuleCard = (container, module) => {
    if (disposed) return;
    selectedCard = container;
    selectedModule = module;
    container?.querySelector(".alexz-module-update")?.remove();
    renderSelectedCard();
    syncDisabled();
  };
  const refresh = async () => {
    if (pending || queued || disposed || state?.phase === "checking") return;
    pending = true;
    syncDisabled();
    try {
      await check();
    } catch (error) {
      if (!disposed) showError(error);
    } finally {
      pending = false;
      if (!disposed) syncDisabled();
    }
  };
  const dispose = () => {
    disposed = true;
    controller.abort();
    if (timer) clearTimeout(timer);
    updateButton.remove();
    unlock();
    selectedCard?.querySelector(".alexz-module-update")?.remove();
    selectedCard?.querySelector(".alexz-module-update-button")?.remove();
    options.alert.classList.remove("alexz-update-summary");
    host.remove();
  };
  void poll().catch((error) => {
    if (!disposed) showError(error);
  });
  return { refresh, dispose, renderModuleCard };
}
export {
  mountModuleUpdates
};
//# sourceMappingURL=module_updates.js.map
