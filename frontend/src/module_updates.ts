/** Dependency-aware update controls for the existing Module Node Picker. */

type ModuleStatus = "safe" | "risk" | "unknown" | "blocked" | "up_to_date";
interface Addition { name: string; version: string }
interface ModulePlan {
  module: string;
  status: ModuleStatus;
  reasons: string[];
  requirements_changed: boolean | null;
  additions: Addition[];
  before?: string;
  target?: string;
  diff: string;
}
interface UpdatePlan {
  id: string;
  expires_at: number;
  results: ModulePlan[];
  counts: Record<ModuleStatus, number>;
  batch_count: number;
  batch_error: string;
  batch: { additions: Addition[] };
  baseline: string[];
}
interface PlanState {
  phase: "idle" | "checking" | "ready" | "error";
  message: string;
  plan: UpdatePlan | null;
  updated_modules?: string[];
  restart_required_modules?: string[];
  execution?: { phase: string; message: string; directory: string };
}
interface Options {
  fetchApi: (route: string, options?: RequestInit) => Promise<Response>;
  refreshButton: HTMLButtonElement;
  alert: HTMLElement;
  alertText: HTMLElement;
  onPlanReady: () => Promise<unknown>;
  onModulesUpdated?: (modules: string[]) => void;
  confirm?: (message: string) => boolean;
}
/** Mount isolated controls, retaining no global state and starting no update on load. */
export function mountModuleUpdates(root: HTMLElement, options: Options): { refresh: () => Promise<void>; dispose: () => void; renderModuleCard: (container: HTMLElement | null, module: string) => void } {
  const document = root.ownerDocument;
  const host = document.createElement("section");
  host.className = "alexz-update-planner";
  host.setAttribute("aria-label", "Обновления модулей с проверкой зависимостей");
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
  let state: PlanState | null = null;
  let selectedCard: HTMLElement | null = null;
  let selectedModule = "";
  let applying = false;
  const frozenControls = new Map<HTMLInputElement | HTMLSelectElement | HTMLButtonElement, boolean>();
  let refreshStarted = false;
  let completedPlanId = "";
  let timer: ReturnType<typeof setTimeout> | null = null;
  const controller = new AbortController();
  const confirm = options.confirm ?? ((message: string) => window.confirm(message));

  /** Create an action with compact failure feedback. */
  function button(parent: HTMLElement, text: string, action: () => Promise<void>): HTMLButtonElement {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "alexz-mod-picker-btn-small";
    node.textContent = text;
    node.onclick = async (event) => {
      event.stopPropagation();
      if (pending || disposed || (queued && node.dataset.cancel !== "1" && node.dataset.status !== "1")) return;
      pending = true;
      syncDisabled();
      try { await action(); }
      catch (error) {
        if (!disposed) showError(error);
      } finally {
        pending = false;
        if (!disposed) syncDisabled();
      }
    };
    parent.append(node);
    return node;
  }

  /** Restore existing control states before applying the next update lock. */
  function unlock(): void {
    root.inert = false;
    root.removeAttribute("aria-busy");
    for (const [node, disabled] of frozenControls) node.disabled = disabled;
    frozenControls.clear();
  }

  /** Freeze the entire picker during an update, including module selection. */
  function syncDisabled(): void {
    unlock();
    options.refreshButton.disabled = pending || queued || state?.phase === "checking";
    updateButton.hidden = state?.phase !== "ready" || !state.plan || pending || queued;
    updateButton.textContent = `Обновить модули (${state?.plan?.batch_count ?? 0})`;
    updateButton.disabled = pending || queued || !state?.plan || state.plan.batch_count === 0
      || Boolean(state.plan.batch_error) || Date.now() / 1000 > state.plan.expires_at;
    updateButton.title = updateButton.disabled ? "Нет проверенных обновлений; подробности в консоли ComfyUI" : "";
    const individual = selectedCard?.querySelector<HTMLButtonElement>(".alexz-module-update-button");
    if (individual) individual.disabled = pending || queued || state?.phase !== "ready" || !state.plan
      || individual.dataset.checked !== "1" || Date.now() / 1000 > state.plan.expires_at;
    if (applying || (queued && state?.execution?.phase !== "waiting_for_shutdown")) {
      for (const node of root.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>("input, select, button, textarea")) {
        frozenControls.set(node, node.disabled);
        node.disabled = true;
      }
      root.inert = true;
      root.setAttribute("aria-busy", "true");
    }
  }

  /** Keep technical error details in the console instead of the picker. */
  function showError(error: unknown): void {
    console.error("[ALEXZ_tools] Обновление модулей:", error);
    summary.textContent = "Ошибка. Подробности в консоли ComfyUI.";
  }

  /** Use ComfyUI's transport and a bounded request with explicit mutation headers. */
  async function request(route: string, payload?: object): Promise<Record<string, unknown>> {
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]);
    const response = await options.fetchApi(`/alexz_tools/update_plan_${route}`, {
      signal, cache: "no-store",
      ...(payload ? { method: "POST", headers: { "Content-Type": "application/json", "X-Alexz-Update": "1" },
        body: JSON.stringify(payload) } : {}),
    });
    if (response.status === 404) throw new Error("Новые backend routes ещё не загружены. Перезапустите ComfyUI.");
    const data = await response.json() as Record<string, unknown>;
    if (!response.ok || data.error) throw new Error(String(data.error ?? `HTTP ${response.status}`));
    return data;
  }

  /** Refresh progress, rescheduling only while this panel is alive. */
  async function poll(): Promise<void> {
    if (timer) clearTimeout(timer);
    const data = await request("status");
    if (disposed) return;
    state = data as unknown as PlanState;
    if (refreshStarted && state.phase === "ready" && state.plan && completedPlanId !== state.plan.id) {
      completedPlanId = state.plan.id;
      pending = true;
      syncDisabled();
      try { await options.onPlanReady(); }
      finally { pending = false; }
      if (disposed) return;
    }
    options.onModulesUpdated?.(state.updated_modules ?? []);
    render();
    if (state.phase === "checking" || (state.execution && !["done", "error", "cancelled"].includes(state.execution.phase))) {
      timer = setTimeout(() => {
        poll().catch((error: Error) => { if (!disposed) showError(error); });
      }, 2000);
    }
  }

  /** Start an explicit network and requirements check without installing anything. */
  async function check(): Promise<void> {
    if (timer) clearTimeout(timer);

    options.alert.style.display = "block";
    summary.textContent = "Проверка upstream и зависимостей…";
    refreshStarted = true;
    await request("check", {});
    await poll();
  }

  /** Execute a checked module or batch while locking all picker interactions. */
  async function apply(module?: ModulePlan): Promise<void> {
    const plan = state?.plan;
    if (!plan || Date.now() / 1000 > plan.expires_at) throw new Error("План устарел. Повторите проверку.");
    const changes = (module?.additions ?? plan.batch.additions).map(item => `${item.name}==${item.version}`).join(", ");
    const selection = module ? module.module : `${plan.batch_count} модулей`;
    if (!confirm(`Обновить ${selection}?\nДобавление пакетов: ${changes || "не требуется"}.\nПосле завершения перезапустите ComfyUI.`)) return;
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

  /** Render only a compact summary, keeping diagnostics in the main console. */
  function render(): void {
    if (!state) return;
    options.alert.style.display = "block";
    results.replaceChildren();
    summary.textContent = state.phase === "checking" ? state.message
      : state.phase === "error" ? "Ошибка проверки. Подробности в консоли ComfyUI."
      : "Проверка ещё не запускалась";
    if (state.plan) {
      const changed = state.plan.results.filter(item => item.before && item.target && item.before !== item.target).length;
      summary.textContent = `Найдено обновлений: ${changed} модулей.`;
    }
    queued = Boolean(state.execution && !["done", "error", "cancelled"].includes(state.execution.phase));
    if (state.execution && !(state.plan && state.execution.phase === "done")) {
      summary.textContent = state.execution.phase === "done" ? "Обновление завершено. Перезапустите ComfyUI."
        : state.execution.phase === "error" ? "Ошибка обновления. Подробности в консоли ComfyUI."
        : state.execution.phase === "cancelled" ? "Обновление отменено."
        : "Обновление выполняется. Ход работы — в консоли ComfyUI.";
      if (state.execution.phase === "waiting_for_shutdown") {
        summary.textContent = "Обновление ожидает остановки ComfyUI.";
        const cancel = button(results, "Отменить ожидающее обновление", async () => {
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

  /** Show only an individual update action when a newer commit was found. */
  function renderSelectedCard(): void {
    selectedCard?.querySelector(".alexz-module-update-button")?.remove();
    if (!selectedCard?.isConnected) return;
    const status = selectedCard.querySelector<HTMLElement>(".alexz-module-runtime-status");
    if (status) {
      if (!status.dataset.originalHtml) {
        status.dataset.originalHtml = status.innerHTML;
        status.dataset.originalClass = status.className;
      }
      if (state?.restart_required_modules?.includes(selectedModule)) {
        status.classList.remove("ok", "unknown");
        status.classList.add("warn");
        const value = status.querySelector("span:last-child");
        if (value) value.textContent = "Требуется перезагрузка";
      } else {
        status.innerHTML = status.dataset.originalHtml;
        status.className = status.dataset.originalClass ?? status.className;
      }
    }
    const item = state?.plan?.results.find(item => item.module === selectedModule);
    if (!item?.before || !item.target || item.before === item.target) return;
    const action = button(selectedCard, "Обновить модуль", () => apply(item));
    action.classList.add("alexz-module-update-button");
    action.dataset.checked = item.status === "safe" ? "1" : "0";
    if (item.status !== "safe") action.title = "Обновление не прошло проверку; подробности в консоли ComfyUI";
  }

  const updateButton = button(options.alert, "Обновить модули (0)", () => apply());
  options.alert.style.display = "block";
  syncDisabled();
  const renderModuleCard = (container: HTMLElement | null, module: string) => {
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
    try { await check(); }
    catch (error) {
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
  void poll().catch((error: Error) => { if (!disposed) showError(error); });
  return { refresh, dispose, renderModuleCard };
}
