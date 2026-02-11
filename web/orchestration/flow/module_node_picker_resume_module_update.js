/**
 * Module: web/orchestration/flow/module_node_picker_resume_module_update.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Resume flow for pending module-update jobs.
 *
 * Purpose:
 *   Restores update poll lifecycle and post-update follow-ups after picker
 *   re-open/re-render without changing existing status semantics.
 */

/**
 * Resume in-flight module-update job after picker re-open/re-render.
 */
export async function resumePendingModuleUpdateFlowImpl(context) {
    const hasPendingUpdate = typeof context?.hasPendingUpdate === "function"
        ? context.hasPendingUpdate
        : () => false;
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const setActionBusy = typeof context?.setActionBusy === "function"
        ? context.setActionBusy
        : () => {};
    const setProcessAction = typeof context?.setProcessAction === "function"
        ? context.setProcessAction
        : () => {};
    const setRefreshLine = typeof context?.setRefreshLine === "function"
        ? context.setRefreshLine
        : () => {};
    const fetchModuleUpdateStatus = typeof context?.fetchModuleUpdateStatus === "function"
        ? context.fetchModuleUpdateStatus
        : async () => ({});
    const setProcessTarget = typeof context?.setProcessTarget === "function"
        ? context.setProcessTarget
        : () => {};
    const formatUpdateLine = typeof context?.formatUpdateLine === "function"
        ? context.formatUpdateLine
        : () => ({ text: "", tone: "neutral" });
    const pollUpdateProgress = typeof context?.pollUpdateProgress === "function"
        ? context.pollUpdateProgress
        : async () => null;
    const clearPendingUpdate = typeof context?.clearPendingUpdate === "function"
        ? context.clearPendingUpdate
        : () => {};
    const maybeInstallChangedRequirements = typeof context?.maybeInstallChangedRequirements === "function"
        ? context.maybeInstallChangedRequirements
        : async () => {};
    const loadCatalog = typeof context?.loadCatalog === "function"
        ? context.loadCatalog
        : async () => {};
    const loadModuleInfo = typeof context?.loadModuleInfo === "function"
        ? context.loadModuleInfo
        : async () => {};
    const isCanceledRequestError = typeof context?.isCanceledRequestError === "function"
        ? context.isCanceledRequestError
        : () => false;

    if (!hasPendingUpdate()) {
        return;
    }
    if (!shouldContinue()) {
        return;
    }
    setActionBusy(true);
    setProcessAction("", "", null);
    setRefreshLine("Resuming module update status...", "neutral");
    try {
        const payload = await fetchModuleUpdateStatus();
        if (!shouldContinue()) {
            return;
        }
        const update = payload?.update || {};
        const scope = String(update?.scope || "").trim().toLowerCase();
        setProcessTarget(scope === "comfyui" ? "comfy" : "custom");
        const line = formatUpdateLine(update);
        setRefreshLine(line.text, line.tone);

        if (Boolean(update?.running)) {
            const done = await pollUpdateProgress();
            if (!shouldContinue()) {
                return;
            }
            if (!done) {
                return;
            }
            if (!Boolean(done?.running) && String(done?.phase || "") !== "starting") {
                clearPendingUpdate();
            }
            if (String(done?.phase || "") === "done") {
                await maybeInstallChangedRequirements(done);
            }
            if (!shouldContinue()) {
                return;
            }
            await loadCatalog();
            if (!shouldContinue()) {
                return;
            }
            await loadModuleInfo();
            return;
        }

        const phase = String(update?.phase || "").trim().toLowerCase();
        if (phase === "done" || phase === "error") {
            if (phase === "done") {
                await maybeInstallChangedRequirements(update);
            }
            if (!shouldContinue()) {
                return;
            }
            await loadCatalog();
            if (!shouldContinue()) {
                return;
            }
            await loadModuleInfo();
            clearPendingUpdate();
            return;
        }

        // Stale marker: no active/terminal update job available.
        clearPendingUpdate();
        setRefreshLine("No pending module update job found.", "neutral");
    } catch (err) {
        if (!shouldContinue()) {
            return;
        }
        const message = String(err || "");
        if (isCanceledRequestError(err)) {
            return;
        }
        setRefreshLine(`Failed to restore update status: ${message}`, "warn");
    } finally {
        if (!shouldContinue()) {
            return;
        }
        setActionBusy(false);
    }
}
