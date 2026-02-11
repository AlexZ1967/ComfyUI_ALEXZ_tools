/**
 * Module: web/orchestration/runtime/module_node_picker_runtime_bootstrap_bindings.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime-bootstrap callback bindings for Module Node Picker composer.
 *
 * Purpose:
 *   Encapsulates callback adapters passed from composer to runtime bootstrap so
 *   composition code remains focused on wiring stages instead of inline lambdas.
 */

import { runModuleNodePickerStartupLoad } from "../core/module_node_picker_bindings.js";

/**
 * Build runtime-bootstrap callback adapters from flow/stage dependencies.
 */
export function createModuleNodePickerRuntimeBootstrapBindings(context = {}) {
    const flowStage = context?.flowStage;
    const stageAdapters = context?.stageAdapters || {};
    const isPickerAlive = typeof context?.isPickerAlive === "function"
        ? context.isPickerAlive
        : () => true;
    const pickerStore = context?.pickerStore;
    const defaultModule = String(context?.defaultModule || "");

    return {
        loadModuleInfo: (options = {}) => stageAdapters.loadModuleInfo?.(options),
        loadCatalog: (options = {}) => stageAdapters.loadCatalog?.(options),
        setExpandedModule: (value) => stageAdapters.setExpandedModule?.(value),
        runModuleUpdate: (...args) => flowStage?.runModuleUpdate?.(...args),
        installComfyUIRequirementsFlow: (...args) => flowStage?.installComfyUIRequirementsFlow?.(...args),
        refreshComfyUIInfoFlow: (...args) => flowStage?.refreshComfyUIInfoFlow?.(...args),
        refreshCustomNodesInfoFlow: (...args) => flowStage?.refreshCustomNodesInfoFlow?.(...args),
        resumePendingCustomRefreshFlow: (...args) => flowStage?.resumePendingCustomRefreshFlow?.(...args),
        resumePendingModuleUpdateFlow: (...args) => flowStage?.resumePendingModuleUpdateFlow?.(...args),
        resumePendingComfyInfoRefreshFlow: (...args) => flowStage?.resumePendingComfyInfoRefreshFlow?.(...args),
        startCatalogStartupLoad: (options = {}) => runModuleNodePickerStartupLoad({
            pickerStore,
            defaultModule,
            loadCatalog: (opts = {}) => stageAdapters.loadCatalog?.(opts),
            shouldContinue: isPickerAlive,
            startupRetries: 2,
            startupRetryDelayMs: 250,
            onSettled: options?.onSettled,
        }),
    };
}
