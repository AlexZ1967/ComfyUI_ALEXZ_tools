/**
 * Module: web/orchestration/flow/module_node_picker_resume_comfy_refresh.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Resume flow for pending ComfyUI-info refresh actions.
 *
 * Purpose:
 *   Restores ComfyUI status-check workflow after picker re-open and keeps
 *   alert/process-card state consistent with regular refresh flow.
 */

/**
 * Resume interrupted ComfyUI info refresh after picker re-open/re-render.
 */
export async function resumePendingComfyInfoRefreshFlowImpl(context) {
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
