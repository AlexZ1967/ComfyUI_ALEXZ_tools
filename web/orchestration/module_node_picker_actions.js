/**
 * Module: web/orchestration/module_node_picker_actions.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   UI action flows for Module Node Picker cards and refresh buttons.
 *
 * Purpose:
 *   Keeps button-driven async workflows outside the main picker render file.
 */

import { shouldContinueContext } from "./runtime/module_node_picker_lifecycle_guard.js";
import { isCanceledRequestError } from "./module_node_picker_error_utils.js";

/**
 * Refresh selected module info and keep result inline in module card.
 */
export async function runRefreshModuleInfoAction(moduleName, syncUpstream, context) {
    const normalized = String(moduleName || "").trim();
    if (!normalized) {
        return;
    }
    if (!shouldContinueContext(context)) {
        return;
    }
    context?.setProcessTarget?.("");
    context?.setRefreshLine?.("", "neutral");
    context?.setProcessAction?.("", "", null);
    context?.setModuleInlineStatus?.(normalized, "Refreshing module info...", "neutral");
    context?.setActionBusy?.(true);
    try {
        await context?.loadModuleInfo?.({
            forceRefresh: true,
            syncUpstream: Boolean(syncUpstream),
            throwOnError: true,
        });
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setModuleInlineStatus?.(normalized, "Module info updated.", "ok");
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setModuleInlineStatus?.(normalized, `Failed to refresh module info: ${String(err)}`, "warn");
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setActionBusy?.(false);
        await context?.loadModuleInfo?.({ forceRefresh: false, syncUpstream: false });
        context?.syncUpdateAllButton?.();
    }
}

/**
 * Install requirements for one custom module and refresh its card state.
 */
export async function runInstallSingleModuleRequirementsAction(moduleName, context) {
    const normalized = String(moduleName || "").trim();
    if (!normalized) {
        return;
    }
    if (!shouldContinueContext(context)) {
        return;
    }
    context?.setProcessTarget?.("custom");
    context?.setRefreshLine?.(`Installing requirements for ${normalized}...`, "neutral");
    context?.setProcessAction?.("", "", null);
    context?.setModuleInlineStatus?.(normalized, "Installing module requirements...", "neutral");
    context?.setActionBusy?.(true);
    try {
        const install = await context?.installModuleRequirements?.([normalized]);
        if (!shouldContinueContext(context)) {
            return;
        }
        const failed = Number(install?.failed || 0);
        const installed = Number(install?.installed || 0);
        if (failed > 0 || installed <= 0) {
            context?.setModuleInlineStatus?.(normalized, "Module requirements install failed.", "warn");
        } else {
            context?.setModuleInlineStatus?.(normalized, "Module requirements installed.", "ok");
        }
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setModuleInlineStatus?.(normalized, `Module requirements install failed: ${String(err)}`, "warn");
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setActionBusy?.(false);
        await context?.loadModuleInfo?.({ forceRefresh: false, syncUpstream: false });
        context?.syncUpdateAllButton?.();
    }
}

/**
 * Refresh ComfyUI info card with forced upstream sync.
 */
export async function runRefreshComfyUIInfoAction(context) {
    if (!shouldContinueContext(context)) {
        return;
    }
    context?.setComfyStatusChecked?.(true);
    context?.setPendingComfyInfoRefresh?.(true);
    context?.setActionBusy?.(true);
    context?.setProcessTarget?.("comfy");
    context?.setProcessAction?.("", "", null);
    context?.setRefreshLine?.("Refreshing ComfyUI info...", "neutral");
    const comfyAlert = context?.comfyAlert;
    const comfyAlertText = context?.comfyAlertText;
    const comfyUpdateBtn = context?.comfyUpdateBtn;
    const comfyInstallReqBtn = context?.comfyInstallReqBtn;
    const logMode = typeof context?.getLogMode === "function" ? context.getLogMode() : "summary";
    if (comfyAlert && comfyAlertText) {
        comfyAlert.style.display = "block";
        comfyAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
        comfyAlertText.textContent = "Refreshing ComfyUI info...";
    }
    try {
        const payload = await context?.fetchComfyUIInfo?.(true, true, context?.getComfyMode?.(), { logMode });
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.renderComfyAlert?.(payload?.comfyui || null);
        context?.clearPendingComfyInfoRefresh?.();
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        if (isCanceledRequestError(err)) {
            return;
        }
        if (comfyAlert && comfyAlertText) {
            comfyAlert.classList.remove(
                "alexz-mod-picker-status-card--warn",
                "alexz-mod-picker-status-card--ok",
                "alexz-mod-picker-status-card--neutral"
            );
            comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
            comfyAlert.style.display = "block";
            comfyAlertText.textContent = `Failed to refresh ComfyUI info: ${String(err)}`;
        }
        if (comfyUpdateBtn) {
            comfyUpdateBtn.style.display = "none";
        }
        if (comfyInstallReqBtn) {
            comfyInstallReqBtn.style.display = "none";
        }
        context?.clearPendingComfyInfoRefresh?.();
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setActionBusy?.(false);
        context?.syncUpdateAllButton?.();
    }
}

/**
 * Refresh custom modules runtime state and reload catalog.
 */
export async function runRefreshCustomNodesInfoAction(context) {
    if (!shouldContinueContext(context)) {
        return;
    }
    context?.setCustomStatusChecked?.(true);
    context?.setPendingCustomRefresh?.(true);
    context?.setActionBusy?.(true);
    context?.setProcessTarget?.("custom");
    context?.setProcessAction?.("", "", null);
    context?.setRefreshLine?.("Refreshing Custom Nodes info...", "neutral");
    const customAlert = context?.customAlert;
    const customAlertText = context?.customAlertText;
    if (customAlert && customAlertText) {
        customAlert.style.display = "block";
        customAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        customAlert.classList.add("alexz-mod-picker-status-card--neutral");
        customAlertText.textContent = "Refreshing Custom Nodes info...";
    }
    try {
        await context?.refreshModuleRuntimeState?.({
            logMode: typeof context?.getLogMode === "function" ? context.getLogMode() : "summary",
        });
        if (!shouldContinueContext(context)) {
            return;
        }
        const ok = await context?.pollRefreshProgress?.();
        if (!shouldContinueContext(context)) {
            return;
        }
        if (!ok) {
            context?.setRefreshLine?.("Custom Nodes refresh finished with errors.", "warn");
        } else {
            try {
                await context?.acknowledgeAllModuleNovelty?.();
                if (!shouldContinueContext(context)) {
                    return;
                }
            } catch (err) {
                if (!shouldContinueContext(context)) {
                    return;
                }
                context?.setRefreshLine?.(`Refresh completed, but novelty reset failed: ${String(err)}`, "warn");
            }
        }
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setRefreshLine?.(`Custom Nodes refresh error: ${String(err)}`, "warn");
    } finally {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setActionBusy?.(false);
    }
    if (!shouldContinueContext(context)) {
        return;
    }
    try {
        await context?.loadCatalog?.();
    } finally {
        context?.clearPendingCustomRefresh?.();
    }
}
