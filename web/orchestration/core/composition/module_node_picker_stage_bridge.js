/**
 * Module: web/orchestration/core/composition/module_node_picker_stage_bridge.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Deferred-stage bridge for Module Node Picker composer.
 *
 * Purpose:
 *   Keeps temporary "deferred stage" handlers and flow-stage wiring in one
 *   place so composer code stays focused on high-level assembly.
 */

/**
 * Create deferred stage bridge used before flow-stage handlers are bound.
 */
export function createModuleNodePickerStageBridge() {
    const deferredStage = {
        loadModuleInfo: async () => {},
        loadCatalog: async () => {},
        renderNodeList: () => {},
        setExpandedModule: () => {},
    };

    const adapters = {
        loadModuleInfo: (options = {}) => deferredStage.loadModuleInfo(options),
        loadCatalog: (options = {}) => deferredStage.loadCatalog(options),
        renderNodeList: () => deferredStage.renderNodeList(),
        setExpandedModule: (value) => deferredStage.setExpandedModule(value),
    };

    const wireFlowStage = (flowStage) => {
        if (!flowStage) {
            return;
        }
        deferredStage.loadModuleInfo = (options = {}) => flowStage.loadModuleInfo(options);
        deferredStage.loadCatalog = (options = {}) => flowStage.loadCatalog(options);
        deferredStage.renderNodeList = () => flowStage.renderNodeList();
        deferredStage.setExpandedModule = (value) => flowStage.setExpandedModule(value);
    };

    return {
        deferredStage,
        adapters,
        wireFlowStage,
    };
}
