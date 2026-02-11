/**
 * Module: web/orchestration/flow/progress/module_node_picker_update_flow.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Refresh/update orchestration helpers for Module Node Picker.
 *
 * Purpose:
 *   Keeps long-running polling and update flows outside the main UI file.
 */

import { shouldContinueContext } from "../../runtime/lifecycle/module_node_picker_lifecycle_guard.js";
import { isCanceledRequestError } from "../../core/infra/module_node_picker_error_utils.js";

/**
 * Poll custom-module refresh status until completion or failure.
 */
export async function pollRefreshProgressLoop(context) {
    const isTokenActive = context?.isTokenActive;
    const fetchModuleRefreshStatus = context?.fetchModuleRefreshStatus;
    const formatRefreshLine = context?.formatRefreshLine;
    const setRefreshLine = context?.setRefreshLine;
    const getProcessTarget = context?.getProcessTarget;
    const customAlert = context?.customAlert;
    const customAlertText = context?.customAlertText;
    const sleepMs = Number(context?.sleepMs || 400);

    if (typeof isTokenActive !== "function" || typeof fetchModuleRefreshStatus !== "function") {
        return false;
    }
    while (isTokenActive() && shouldContinueContext(context)) {
        let payload;
        try {
            payload = await fetchModuleRefreshStatus();
        } catch (err) {
            if (!shouldContinueContext(context)) {
                return false;
            }
            if (isCanceledRequestError(err)) {
                return false;
            }
            setRefreshLine?.(`Custom Nodes refresh status failed (${String(err)}).`, "warn");
            return false;
        }
        if (!shouldContinueContext(context)) {
            return false;
        }
        const refresh = payload?.refresh || {};
        const line = typeof formatRefreshLine === "function"
            ? formatRefreshLine(refresh)
            : { text: String(refresh?.message || ""), tone: "neutral" };
        setRefreshLine?.(line.text, line.tone);
        if (typeof getProcessTarget === "function" && getProcessTarget() === "custom" && customAlert && customAlertText) {
            customAlert.style.display = "block";
            customAlert.classList.remove(
                "alexz-mod-picker-status-card--warn",
                "alexz-mod-picker-status-card--ok",
                "alexz-mod-picker-status-card--neutral"
            );
            if (line.tone === "warn") {
                customAlert.classList.add("alexz-mod-picker-status-card--warn");
            } else if (line.tone === "ok") {
                customAlert.classList.add("alexz-mod-picker-status-card--ok");
            } else {
                customAlert.classList.add("alexz-mod-picker-status-card--neutral");
            }
            customAlertText.textContent = String(line.text || "");
        }
        if (!refresh?.running) {
            return refresh?.phase !== "error";
        }
        await new Promise((resolve) => setTimeout(resolve, sleepMs));
    }
    return false;
}

/**
 * Poll module-update status until completion or failure.
 */
export async function pollUpdateProgressLoop(context) {
    const isTokenActive = context?.isTokenActive;
    const fetchModuleUpdateStatus = context?.fetchModuleUpdateStatus;
    const formatUpdateLine = context?.formatUpdateLine;
    const setRefreshLine = context?.setRefreshLine;
    const sleepMs = Number(context?.sleepMs || 450);

    if (typeof isTokenActive !== "function" || typeof fetchModuleUpdateStatus !== "function") {
        return null;
    }
    while (isTokenActive() && shouldContinueContext(context)) {
        let payload;
        try {
            payload = await fetchModuleUpdateStatus();
        } catch (err) {
            if (!shouldContinueContext(context)) {
                return null;
            }
            if (isCanceledRequestError(err)) {
                return null;
            }
            setRefreshLine?.(`Update status failed (${String(err)}).`, "warn");
            return null;
        }
        if (!shouldContinueContext(context)) {
            return null;
        }
        const update = payload?.update || {};
        const line = typeof formatUpdateLine === "function"
            ? formatUpdateLine(update)
            : { text: String(update?.message || ""), tone: "neutral" };
        setRefreshLine?.(line.text, line.tone);
        if (!update?.running) {
            return update;
        }
        await new Promise((resolve) => setTimeout(resolve, sleepMs));
    }
    return null;
}

/**
 * Install ComfyUI requirements and refresh ComfyUI status card.
 */
export async function runInstallComfyUIRequirementsFlow(context) {
    const setActionBusy = context?.setActionBusy;
    const setProcessTarget = context?.setProcessTarget;
    const setRefreshLine = context?.setRefreshLine;
    const installComfyUIRequirements = context?.installComfyUIRequirements;
    const fetchComfyUIInfo = context?.fetchComfyUIInfo;
    const getComfyMode = context?.getComfyMode;
    const renderComfyAlert = context?.renderComfyAlert;
    const setProcessAction = context?.setProcessAction;
    const syncUpdateAllButton = context?.syncUpdateAllButton;

    if (!shouldContinueContext(context)) {
        return;
    }
    setActionBusy?.(true);
    setProcessTarget?.("comfy");
    try {
        if (!shouldContinueContext(context)) {
            return;
        }
        setRefreshLine?.("Installing ComfyUI dependencies (pip)...", "neutral");
        const install = await installComfyUIRequirements?.();
        if (!shouldContinueContext(context)) {
            return;
        }
        if (String(install?.status || "") !== "installed") {
            setRefreshLine?.("ComfyUI dependencies install failed.", "warn");
            return;
        }
        const comfyPayload = await fetchComfyUIInfo?.(false, false, getComfyMode?.());
        if (!shouldContinueContext(context)) {
            return;
        }
        renderComfyAlert?.(comfyPayload?.comfyui || null);
        setRefreshLine?.("ComfyUI dependencies installed.", "ok");
        setProcessAction?.("", "", null);
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        setActionBusy?.(false);
        syncUpdateAllButton?.();
    }
}

/**
 * Handle requirements-install prompts after update completion.
 */
export async function maybeInstallChangedRequirementsFlow(update, context) {
    if (!shouldContinueContext(context)) {
        return;
    }
    const setRefreshLine = context?.setRefreshLine;
    const setProcessAction = context?.setProcessAction;
    const installComfyUIRequirementsFlow = context?.installComfyUIRequirementsFlow;
    const installModuleRequirements = context?.installModuleRequirements;
    const setActionBusy = context?.setActionBusy;

    const scope = String(update?.scope || "");
    if (scope === "comfyui") {
        if (!Boolean(update?.requirements_changed)) {
            return;
        }
        setRefreshLine?.("ComfyUI requirements.txt changed. Install dependencies?", "warn");
        setProcessAction?.(
            "ComfyUI requirements were updated after pull.",
            "Install ComfyUI requirements",
            installComfyUIRequirementsFlow
        );
        return;
    }

    const modules = Array.isArray(update?.requirements_modules) ? update.requirements_modules : [];
    if (!modules.length) {
        return;
    }
    setRefreshLine?.("Custom module requirements changed. Install dependencies?", "warn");
    setProcessAction?.(
        `requirements.txt changed for: ${modules.join(", ")}.`,
        "Install updated requirements",
        async () => {
            setActionBusy?.(true);
            try {
                setRefreshLine?.("Installing updated dependencies (pip)...", "neutral");
                const install = await installModuleRequirements?.(modules);
                const failed = Number(install?.failed || 0);
                const installed = Number(install?.installed || 0);
                if (failed > 0) {
                    setRefreshLine?.(`Dependencies install finished with errors: ok=${installed}, failed=${failed}.`, "warn");
                    return;
                }
                setRefreshLine?.(`Dependencies installed: ${installed} module(s).`, "ok");
                setProcessAction?.("", "", null);
            } finally {
                setActionBusy?.(false);
            }
        }
    );
}

/**
 * Run module update flow (start + poll + post-actions + UI reload).
 */
export async function runModuleUpdateFlow(scope, moduleName, context) {
    const setActionBusy = context?.setActionBusy;
    const setProcessTarget = context?.setProcessTarget;
    const setProcessAction = context?.setProcessAction;
    const setRefreshLine = context?.setRefreshLine;
    const startModuleUpdate = context?.startModuleUpdate;
    const getLogMode = typeof context?.getLogMode === "function"
        ? context.getLogMode
        : () => "summary";
    const pollUpdateProgress = context?.pollUpdateProgress;
    const getSelectedGroup = context?.getSelectedGroup;
    const getSelectedModule = context?.getSelectedModule;
    const onMarkUpdatedModule = context?.onMarkUpdatedModule;
    const isModuleMarkedUpdated = context?.isModuleMarkedUpdated;
    const maybeInstallChangedRequirements = context?.maybeInstallChangedRequirements;
    const loadCatalog = context?.loadCatalog;
    const loadModuleInfo = context?.loadModuleInfo;
    const syncUpdateAllButton = context?.syncUpdateAllButton;
    const setPendingUpdate = context?.setPendingUpdate;
    const clearPendingUpdate = context?.clearPendingUpdate;
    let pendingRequirementsPrompt = null;

    if (!shouldContinueContext(context)) {
        return;
    }
    setPendingUpdate?.(true);
    setActionBusy?.(true);
    try {
        if (!shouldContinueContext(context)) {
            return;
        }
        if (String(scope || "") === "comfyui") {
            setProcessTarget?.("comfy");
        } else {
            setProcessTarget?.("custom");
        }
        setProcessAction?.("", "", null);
        setRefreshLine?.("Starting update...", "neutral");
        await startModuleUpdate?.(scope, moduleName, { logMode: getLogMode() });
        if (!shouldContinueContext(context)) {
            return;
        }
        const update = await pollUpdateProgress?.();
        if (!shouldContinueContext(context)) {
            return;
        }
        if (!update) {
            return;
        }
        if (!Boolean(update?.running) && String(update?.phase || "") !== "starting") {
            clearPendingUpdate?.();
        }
        const currentGroup = String(getSelectedGroup?.() || "");
        const currentModule = String(getSelectedModule?.() || "").trim();
        const updatedNow = Array.isArray(update?.results)
            ? update.results.filter((item) => String(item?.status || "") === "updated")
            : [];
        for (const item of updatedNow) {
            const mod = String(item?.module || "").trim();
            if (mod) {
                onMarkUpdatedModule?.(mod);
            }
        }
        if (String(update.phase || "") === "done") {
            pendingRequirementsPrompt = update;
        }
        let preferredGroup = currentGroup;
        let preferredModule = currentModule;
        let autoExpandModule = "";
        if (scope === "single") {
            preferredGroup = "custom";
            preferredModule = String(moduleName || currentModule || "").trim();
            if (updatedNow.some((item) => String(item?.module || "").trim() === preferredModule)) {
                autoExpandModule = preferredModule;
            }
        } else if (scope === "all" && currentGroup === "custom" && currentModule) {
            preferredGroup = "custom";
            preferredModule = currentModule;
            if (typeof isModuleMarkedUpdated === "function" && isModuleMarkedUpdated(currentModule)) {
                autoExpandModule = currentModule;
            }
        }
        await loadCatalog?.({ preferredGroup, preferredModule, autoExpandModule });
        if (!shouldContinueContext(context)) {
            return;
        }
        await loadModuleInfo?.();
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        clearPendingUpdate?.();
        setRefreshLine?.(`Update failed (${String(err)}).`, "warn");
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        setActionBusy?.(false);
        syncUpdateAllButton?.();
    }
    if (!shouldContinueContext(context)) {
        return;
    }
    if (pendingRequirementsPrompt) {
        // Show post-update requirements action only after busy-lock is released,
        // otherwise the first click may be ignored while controls are still disabled.
        await maybeInstallChangedRequirements?.(pendingRequirementsPrompt);
    }
}
