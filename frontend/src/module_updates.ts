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
  execution?: { phase: string; message: string; directory: string };
}
interface Options {
  fetchApi: (route: string, options?: RequestInit) => Promise<Response>;
  refreshButton: HTMLButtonElement;
  alert: HTMLElement;
  alertText: HTMLElement;
  onPlanReady: () => Promise<unknown>;
  confirm?: (message: string) => boolean;
}
const labels: Record<ModuleStatus, string> = {
  safe: "🟢 Конфликтов не обнаружено",
  risk: "🔴 Требуется изменение окружения или отдельная проверка",
  unknown: "🟡 Проверка не завершена",
  blocked: "⛔ Git-обновление заблокировано",
  up_to_date: "✓ Уже актуален",
};

/** Mount isolated controls, retaining no global state and starting no update on load. */
export function mountModuleUpdates(root: HTMLElement, options: Options): { refresh: () => Promise<void>; dispose: () => void } {
  const document = root.ownerDocument;
  const host = document.createElement("section");
  host.className = "alexz-update-planner";
  host.setAttribute("aria-label", "Обновления модулей с проверкой зависимостей");
  host.hidden = true;
  const title = document.createElement("strong");
  title.textContent = "Обновления модулей";
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
  let state: PlanState | null = null;
  let expanded = false;
  let refreshStarted = false;
  let completedPlanId = "";
  let timer: ReturnType<typeof setTimeout> | null = null;
  const controller = new AbortController();
  const confirm = options.confirm ?? ((message: string) => window.confirm(message));

  /** Create a text-only action; errors remain visible in the panel. */
  function button(parent: HTMLElement, text: string, action: () => Promise<void>): HTMLButtonElement {
    const node = document.createElement("button");
    node.type = "button";
    node.className = "alexz-mod-picker-btn-small";
    node.textContent = text;
    node.onclick = async () => {
      if (pending || disposed || (queued && node.dataset.cancel !== "1" && node.dataset.details !== "1")) return;
      pending = true;
      syncDisabled();
      try { await action(); }
      catch (error) {
        if (!disposed) summary.textContent = error instanceof Error ? error.message : String(error);
      } finally {
        pending = false;
        if (!disposed) syncDisabled();
      }
    };
    parent.append(node);
    return node;
  }

  /** Prevent duplicate actions and execution of an expired plan. */
  function syncDisabled(): void {
    for (const node of host.querySelectorAll<HTMLButtonElement>("button")) {
      node.disabled = pending || (queued && node.dataset.cancel !== "1") || state?.phase === "checking"
        || (node.dataset.apply === "1" && (!state?.plan || Date.now() / 1000 > state.plan.expires_at));
    }
    options.refreshButton.disabled = pending || queued || state?.phase === "checking";
    detailsButton.disabled = pending || state?.phase === "checking";
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
    render();
    if (state.phase === "checking" || state.execution?.phase === "waiting_for_shutdown") {
      timer = setTimeout(() => {
        poll().catch((error: Error) => { if (!disposed) summary.textContent = error.message; });
      }, 2000);
    }
  }

  /** Start an explicit network and requirements check without installing anything. */
  async function check(): Promise<void> {
    if (timer) clearTimeout(timer);
    expanded = false;
    host.hidden = true;
    detailsButton.setAttribute("aria-expanded", "false");
    options.alert.style.display = "block";
    summary.textContent = "Проверка upstream и зависимостей…";
    refreshStarted = true;
    await request("check", {});
    await poll();
  }

  /** Queue an acknowledged immutable plan; package operations wait for shutdown. */
  async function apply(module: ModulePlan | null, mode: "checked" | "code_only"): Promise<void> {
    const plan = state?.plan;
    if (!plan || Date.now() / 1000 > plan.expires_at) throw new Error("План устарел. Повторите проверку.");
    const selection = module ? module.module : `${plan.batch_count} модулей`;
    const changes = (module ? module.additions : plan.batch.additions).map(item => `${item.name}==${item.version}`).join(", ");
    const risk = mode === "code_only"
      ? "Обновится только код. Зависимости сохранятся; модуль может перестать загружаться.\n" : "";
    const message = `${risk}Поставить обновление ${selection} в очередь?\n`
      + (mode === "checked" ? `Добавление пакетов: ${changes || "не требуется"}.\n` : "")
      + "Затем остановите ComfyUI и дождитесь phase=done в status.json перед запуском. "
      + (mode === "checked" && changes ? "До установки пакетов будет создана резервная копия окружения." : "");
    if (!confirm(message)) return;
    const response = await request("apply", { plan_id: plan.id, module: module?.module,
      mode, confirmed: true, acknowledge_risk: mode === "code_only" });
    queued = true;
    summary.textContent = `${String(response.message)}\nДиагностика: ${String(response.directory)}`;
    syncDisabled();
    await poll();
  }

  /** Render counts and per-module explanations using textContent only. */
  function render(): void {
    if (!state) return;
    options.alert.style.display = "block";
    summary.textContent = state.message;
    detailedSummary.textContent = state.message;
    results.replaceChildren();
    const plan = state.plan;
    if (plan) {
      const changed = plan.results.filter(item => item.before && item.target && item.before !== item.target).length;
      summary.textContent = `Найдено обновлений: ${changed} модулей.`;
      detailedSummary.textContent += `\nПроверенных для общего апдейта: ${plan.batch_count}. `
        + `С риском: ${plan.counts.risk}. Не проверено: ${plan.counts.unknown}. `
        + `Заблокировано: ${plan.counts.blocked}. Актуально: ${plan.counts.up_to_date}.`;
      if (plan.batch_error) detailedSummary.textContent += `\nОбщий набор несовместим: ${plan.batch_error}`;
      if (plan.batch_count > 0) {
        const all = button(results, `Обновить проверенные (${plan.batch_count})`, () => apply(null, "checked"));
        all.dataset.apply = "1";
      }
      if (plan.baseline.length) {
        const old = document.createElement("details");
        const label = document.createElement("summary");
        label.textContent = `Исходные проблемы окружения (${plan.baseline.length})`;
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
          commits.textContent = `${item.before?.slice(0, 10)} → ${item.target.slice(0, 10)}`;
          card.append(commits);
        }
        if (item.requirements_changed !== null) {
          const dependency = document.createElement("div");
          dependency.textContent = item.requirements_changed ? "Requirements изменились" : "Requirements не изменились";
          card.append(dependency);
        }
        if (item.diff) {
          const diff = document.createElement("pre");
          diff.textContent = item.diff;
          card.append(diff);
        }
        if (item.additions.length) {
          const packages = document.createElement("div");
          packages.textContent = `Добавятся: ${item.additions.map(pkg => `${pkg.name}==${pkg.version}`).join(", ")}`;
          card.append(packages);
        }
        if (item.status === "safe") {
          const checked = button(card, "Апдейт", () => apply(item, "checked"));
          checked.dataset.apply = "1";
        }
        if ((item.status === "risk" || item.status === "unknown") && item.before && item.target) {
          const risky = button(card, "Апдейт только кода…", () => apply(item, "code_only"));
          risky.dataset.apply = "1";
        }
        results.append(card);
      }
    }
    if (state.execution) {
      summary.textContent = state.execution.phase === "waiting_for_shutdown"
        ? "Обновление ожидает остановки ComfyUI."
        : `Обновление: ${state.execution.phase}.`;
      detailedSummary.textContent += `\n${state.execution.phase}: ${state.execution.message}\n${state.execution.directory}`;
      queued = !["done", "error", "cancelled"].includes(state.execution.phase);
      if (state.execution.phase === "waiting_for_shutdown" && plan) {
        const cancel = button(results, "Отменить ожидающее обновление", async () => {
          await request("cancel", { plan_id: plan.id });
          await poll();
        });
        cancel.dataset.cancel = "1";
      }
    }
    syncDisabled();
  }

  const detailsButton = button(options.alert, "Подробнее", async () => {
    if (!state) await poll();
    expanded = !expanded;
    host.hidden = !expanded;
    detailsButton.setAttribute("aria-expanded", String(expanded));
  });
  detailsButton.dataset.details = "1";
  detailsButton.classList.add("alexz-update-details-button");
  detailsButton.setAttribute("aria-expanded", "false");
  button(actions, "Статус проверки / обновления", poll);

  const refresh = async () => {
    if (pending || queued || disposed || state?.phase === "checking") return;
    pending = true;
    syncDisabled();
    try { await check(); }
    catch (error) {
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
