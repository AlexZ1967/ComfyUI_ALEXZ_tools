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

/**
 * Return true while current picker/action context is still valid.
 */
function shouldContinueContext(context) {
    const fn = context?.shouldContinue;
    if (typeof fn !== "function") {
        return true;
    }
    try {
        return fn() !== false;
    } catch (_err) {
        return false;
    }
}

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
        await context?.loadModuleInfo?.({ forceRefresh: false, syncUpstream: false });
        context?.setActionBusy?.(false);
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
        await context?.loadModuleInfo?.({ forceRefresh: false, syncUpstream: false });
        context?.setActionBusy?.(false);
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
    context?.setActionBusy?.(true);
    context?.setProcessTarget?.("comfy");
    context?.setProcessAction?.("", "", null);
    context?.setRefreshLine?.("Refreshing ComfyUI info...", "neutral");
    const comfyAlert = context?.comfyAlert;
    const comfyAlertText = context?.comfyAlertText;
    const comfyUpdateBtn = context?.comfyUpdateBtn;
    const comfyInstallReqBtn = context?.comfyInstallReqBtn;
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
        const payload = await context?.fetchComfyUIInfo?.(true, true, context?.getComfyMode?.());
        if (!shouldContinueContext(context)) {
            return;
        }
        context?.renderComfyAlert?.(payload?.comfyui || null);
    } catch (err) {
        if (!shouldContinueContext(context)) {
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
        await context?.refreshModuleRuntimeState?.();
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
    await context?.loadCatalog?.();
}
