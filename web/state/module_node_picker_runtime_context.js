/**
 * Module: web/state/module_node_picker_runtime_context.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime context initializer for Module Node Picker frontend.
 *
 * Purpose:
 *   Builds store/diagnostics/runtime-state accessors in one place and keeps
 *   picker composition focused on orchestration and rendering concerns.
 */

import { createModuleNodePickerStore } from "./store.js";
import { createModuleDiagnosticsLogger } from "../diagnostics/logger.js";
import {
    getRuntimePickerState,
    clearLegacyPersistentFlags,
    createRuntimeStatusAccessors,
    loadComfyCheckMode,
    saveComfyCheckMode,
} from "./module_node_picker_runtime_state.js";

/**
 * Create runtime context object for picker instance.
 */
export function createModuleNodePickerRuntimeContext(context = {}) {
    const windowObj = context?.windowObj || window;
    const defaultModule = context?.defaultModule || "ComfyUI_ALEXZ_tools";
    const keys = context?.keys || {};

    const pickerStore = createModuleNodePickerStore({
        defaultSelectedGroup: "custom",
        defaultSelectedModule: defaultModule,
        defaultDebugEnabled: Boolean(windowObj[keys.debugRuntimeKey]),
        selectedGroupStorageKey: keys.selectedGroupStorageKey,
        selectedModuleStorageKey: keys.selectedModuleStorageKey,
        debugStorageKey: keys.debugStorageKey,
    });

    const diagnosticsLogger = createModuleDiagnosticsLogger({
        namespace: "ALEXZ_tools Node Picker",
        maxEntries: 200,
        debugEnabled: Boolean(pickerStore.get("debugEnabled")),
    });

    const runtimePickerState = getRuntimePickerState(windowObj, keys.runtimeStateKey);
    clearLegacyPersistentFlags(windowObj, {
        customStatusCheckedKey: keys.legacyCustomStatusCheckedKey,
        pendingCustomRefreshKey: keys.legacyPendingCustomRefreshKey,
        pendingUpdateKey: keys.legacyPendingUpdateKey,
    });

    // On first picker open after ComfyUI start, force a deterministic default
    // selection (Custom_Nodes + project module) regardless of stale localStorage.
    if (!runtimePickerState.__selection_initialized__) {
        pickerStore.set({
            selectedGroup: "custom",
            selectedModule: defaultModule,
        });
        runtimePickerState.__selection_initialized__ = true;
    }

    const runtimeStatus = createRuntimeStatusAccessors(runtimePickerState);
    const comfyCheckMode = loadComfyCheckMode(windowObj, keys.comfyCheckModeStorageKey);
    const persistComfyMode = (mode) => {
        saveComfyCheckMode(windowObj, keys.comfyCheckModeStorageKey, mode);
    };

    return {
        pickerStore,
        diagnosticsLogger,
        runtimeStatus,
        comfyCheckMode,
        saveComfyCheckMode: persistComfyMode,
    };
}
