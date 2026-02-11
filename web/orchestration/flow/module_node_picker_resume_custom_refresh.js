/**
 * Module: web/orchestration/flow/module_node_picker_resume_custom_refresh.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Resume flow for pending "Refresh Custom Nodes Info" actions.
 *
 * Purpose:
 *   Restores refresh progress/status after picker re-open while keeping
 *   lifecycle guards and UI state transitions deterministic.
 */

/**
 * Resume in-flight Custom Nodes refresh after picker re-open/re-render.
 */
export async function resumePendingCustomRefreshFlowImpl(context) {
    const hasPendingCustomRefresh = typeof context?.hasPendingCustomRefresh === "function"
        ? context.hasPendingCustomRefresh
        : () => false;
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const setCustomStatusChecked = typeof context?.setCustomStatusChecked === "function"
        ? context.setCustomStatusChecked
        : () => {};
    const setActionBusy = typeof context?.setActionBusy === "function"
        ? context.setActionBusy
        : () => {};
    const setProcessTarget = typeof context?.setProcessTarget === "function"
        ? context.setProcessTarget
        : () => {};
    const setProcessAction = typeof context?.setProcessAction === "function"
        ? context.setProcessAction
        : () => {};
    const setRefreshLine = typeof context?.setRefreshLine === "function"
        ? context.setRefreshLine
        : () => {};
    const setCustomRefreshCardLine = typeof context?.setCustomRefreshCardLine === "function"
        ? context.setCustomRefreshCardLine
        : () => {};
    const fetchModuleRefreshStatus = typeof context?.fetchModuleRefreshStatus === "function"
        ? context.fetchModuleRefreshStatus
        : async () => ({});
    const pollRefreshProgress = typeof context?.pollRefreshProgress === "function"
        ? context.pollRefreshProgress
        : async () => false;
    const acknowledgeAllModuleNovelty = typeof context?.acknowledgeAllModuleNovelty === "function"
        ? context.acknowledgeAllModuleNovelty
        : async () => {};
    const loadCatalog = typeof context?.loadCatalog === "function"
        ? context.loadCatalog
        : async () => {};
    const clearPendingCustomRefresh = typeof context?.clearPendingCustomRefresh === "function"
        ? context.clearPendingCustomRefresh
        : () => {};
    const isCanceledRequestError = typeof context?.isCanceledRequestError === "function"
        ? context.isCanceledRequestError
        : () => false;

    if (!hasPendingCustomRefresh()) {
        return;
    }
    if (!shouldContinue()) {
        return;
    }

    setCustomStatusChecked(true);
    setActionBusy(true);
    setProcessTarget("custom");
    setProcessAction("", "", null);
    setRefreshLine("Resuming Custom Nodes refresh status...", "neutral");
    setCustomRefreshCardLine("Resuming Custom Nodes refresh status...", "neutral");
    try {
        const payload = await fetchModuleRefreshStatus();
        if (!shouldContinue()) {
            return;
        }
        const refresh = payload?.refresh || {};
        const line = context?.formatRefreshLine
            ? context.formatRefreshLine(refresh)
            : { text: "", tone: "neutral" };
        setRefreshLine(line.text, line.tone);
        setCustomRefreshCardLine(line.text, line.tone);
        if (Boolean(refresh?.running)) {
            const ok = await pollRefreshProgress();
            if (!shouldContinue()) {
                return;
            }
            if (!ok) {
                setRefreshLine("Custom Nodes refresh finished with errors.", "warn");
                setCustomRefreshCardLine("Custom Nodes refresh finished with errors.", "warn");
            } else {
                try {
                    await acknowledgeAllModuleNovelty();
                } catch (err) {
                    if (shouldContinue()) {
                        const message = `Refresh completed, but novelty reset failed: ${String(err)}`;
                        setRefreshLine(message, "warn");
                        setCustomRefreshCardLine(message, "warn");
                    }
                }
            }
            if (!shouldContinue()) {
                return;
            }
            await loadCatalog();
            if (!shouldContinue()) {
                return;
            }
            clearPendingCustomRefresh();
            return;
        }
        if (String(refresh?.phase || "") === "done") {
            try {
                await acknowledgeAllModuleNovelty();
            } catch (err) {
                if (shouldContinue()) {
                    const message = `Refresh completed, but novelty reset failed: ${String(err)}`;
                    setRefreshLine(message, "warn");
                    setCustomRefreshCardLine(message, "warn");
                }
            }
        }
        if (String(refresh?.phase || "") === "done" || String(refresh?.phase || "") === "error") {
            await loadCatalog();
        }
        clearPendingCustomRefresh();
    } catch (err) {
        if (!shouldContinue()) {
            return;
        }
        const message = String(err || "");
        if (isCanceledRequestError(err)) {
            return;
        }
        const line = `Failed to restore refresh status: ${message}`;
        setRefreshLine(line, "warn");
        setCustomRefreshCardLine(line, "warn");
    } finally {
        if (!shouldContinue()) {
            return;
        }
        setActionBusy(false);
    }
}
