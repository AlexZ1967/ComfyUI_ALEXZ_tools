/**
 * Module: web/orchestration/flow/actions/module_node_picker_actions.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   UI action flows for Module Node Picker cards and refresh buttons.
 *
 * Purpose:
 *   Keeps button-driven async workflows outside the main picker render file.
 */

import { shouldContinueContext } from "../../runtime/lifecycle/module_node_picker_lifecycle_guard.js";
import { isCanceledRequestError } from "../../core/infra/module_node_picker_error_utils.js";

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
 * Request manual requirements instructions for one custom module and refresh its card state.
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
    context?.setRefreshLine?.(`Loading manual dependency instructions for ${normalized}...`, "neutral");
    context?.setProcessAction?.("", "", null);
    context?.setModuleInlineStatus?.(normalized, "Manual dependency install required.", "warn");
    context?.setActionBusy?.(true);
    try {
        const advisory = await context?.installModuleRequirements?.([normalized]);
        if (!shouldContinueContext(context)) {
            return;
        }
        if (String(advisory?.status || "") !== "advisory") {
            context?.setModuleInlineStatus?.(normalized, "Failed to load manual dependency instructions.", "warn");
        } else {
            const commands = Array.isArray(advisory?.commands)
                ? advisory.commands.map((item) => String(item || "").trim()).filter(Boolean)
                : [];
            context?.setProcessAction?.(
                commands.length > 0
                    ? `Run manually in the ComfyUI Python environment: ${commands.join(" ; ")}`
                    : "Run dependency install manually in the ComfyUI Python environment or use ComfyUI-Manager.",
                "",
                null
            );
            context?.setRefreshLine?.(`Manual dependency install required for ${normalized}.`, "warn");
        }
    } catch (err) {
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.setModuleInlineStatus?.(normalized, `Failed to load manual dependency instructions: ${String(err)}`, "warn");
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
    const finalizeUiState = () => {
        if (typeof context?.resetBusyState === "function") {
            context.resetBusyState(true);
        } else {
            if (!shouldContinueContext(context)) {
                return;
            }
            context?.setActionBusy?.(false);
            context?.setCatalogControlsLoading?.(false);
            context?.syncBusyUiState?.();
        }
        context?.syncUpdateAllButton?.();
    };

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
                context?.clearUpdatedModulesSession?.();
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
        finalizeUiState();
    }
    if (!shouldContinueContext(context)) {
        return;
    }
    try {
        await context?.loadCatalog?.();
    } finally {
        context?.clearPendingCustomRefresh?.();
        finalizeUiState();
    }
}
