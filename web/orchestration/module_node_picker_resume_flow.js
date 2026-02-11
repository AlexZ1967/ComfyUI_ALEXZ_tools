/**
 * Module: web/orchestration/module_node_picker_resume_flow.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Resume flows for pending refresh/update actions after picker re-open.
 *
 * Purpose:
 *   Keeps restore behavior deterministic across tab switches and re-renders
 *   while preserving original UI status/progress semantics.
 */

/**
 * Resume in-flight Custom Nodes refresh after picker re-open/re-render.
 */
export async function resumePendingCustomRefreshFlow(context) {
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

/**
 * Resume in-flight module-update job after picker re-open/re-render.
 */
export async function resumePendingModuleUpdateFlow(context) {
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

/**
 * Resume interrupted ComfyUI info refresh after picker re-open/re-render.
 */
export async function resumePendingComfyInfoRefreshFlow(context) {
    const hasPendingComfyInfoRefresh = typeof context?.hasPendingComfyInfoRefresh === "function"
        ? context.hasPendingComfyInfoRefresh
        : () => false;
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
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
    const fetchComfyUIInfo = typeof context?.fetchComfyUIInfo === "function"
        ? context.fetchComfyUIInfo
        : async () => ({});
    const getComfyMode = typeof context?.getComfyMode === "function"
        ? context.getComfyMode
        : () => "releases";
    const getLogMode = typeof context?.getLogMode === "function"
        ? context.getLogMode
        : () => "summary";
    const renderComfyAlert = typeof context?.renderComfyAlert === "function"
        ? context.renderComfyAlert
        : () => {};
    const clearPendingComfyInfoRefresh = typeof context?.clearPendingComfyInfoRefresh === "function"
        ? context.clearPendingComfyInfoRefresh
        : () => {};
    const syncUpdateAllButton = typeof context?.syncUpdateAllButton === "function"
        ? context.syncUpdateAllButton
        : () => {};
    const isCanceledRequestError = typeof context?.isCanceledRequestError === "function"
        ? context.isCanceledRequestError
        : () => false;
    const comfyAlert = context?.comfyAlert || null;
    const comfyAlertText = context?.comfyAlertText || null;

    if (!hasPendingComfyInfoRefresh()) {
        return;
    }
    if (!shouldContinue()) {
        return;
    }
    setActionBusy(true);
    setProcessTarget("comfy");
    setProcessAction("", "", null);
    setRefreshLine("Resuming ComfyUI info refresh...", "neutral");
    if (comfyAlert && comfyAlertText) {
        comfyAlert.style.display = "block";
        comfyAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
        comfyAlertText.textContent = "Resuming ComfyUI info refresh...";
    }
    try {
        const payload = await fetchComfyUIInfo(true, true, getComfyMode(), { logMode: getLogMode() });
        if (!shouldContinue()) {
            return;
        }
        renderComfyAlert(payload?.comfyui || null);
        clearPendingComfyInfoRefresh();
    } catch (err) {
        if (!shouldContinue()) {
            return;
        }
        const message = String(err || "");
        if (isCanceledRequestError(err)) {
            return;
        }
        if (comfyAlert && comfyAlertText) {
            comfyAlert.style.display = "block";
            comfyAlert.classList.remove(
                "alexz-mod-picker-status-card--warn",
                "alexz-mod-picker-status-card--ok",
                "alexz-mod-picker-status-card--neutral"
            );
            comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
            comfyAlertText.textContent = `Failed to restore ComfyUI info refresh: ${message}`;
        }
        clearPendingComfyInfoRefresh();
    } finally {
        if (!shouldContinue()) {
            return;
        }
        setActionBusy(false);
        syncUpdateAllButton();
    }
}
