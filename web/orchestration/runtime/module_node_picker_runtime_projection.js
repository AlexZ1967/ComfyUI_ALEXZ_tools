/**
 * Module: web/orchestration/runtime/module_node_picker_runtime_projection.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime-setup projection helpers for Module Node Picker composer.
 *
 * Purpose:
 *   Keeps runtimeSetup unpacking in one place so composition code stays focused
 *   on stage wiring and lifecycle flow.
 */

/**
 * Project runtime setup object into flat named handles used by composer.
 */
export function projectModuleNodePickerRuntimeSetup(runtimeSetup = {}) {
    const runtimeStatus = runtimeSetup?.runtimeStatus || {};

    return {
        pickerStore: runtimeSetup?.pickerStore,
        diagnosticsLogger: runtimeSetup?.diagnosticsLogger,
        loadCustomStatusChecked: runtimeStatus?.loadCustomStatusChecked,
        saveCustomStatusChecked: runtimeStatus?.saveCustomStatusChecked,
        loadComfyStatusChecked: runtimeStatus?.loadComfyStatusChecked,
        saveComfyStatusChecked: runtimeStatus?.saveComfyStatusChecked,
        loadComfyInfoSnapshot: runtimeStatus?.loadComfyInfoSnapshot,
        saveComfyInfoSnapshot: runtimeStatus?.saveComfyInfoSnapshot,
        hasPendingCustomRefresh: runtimeStatus?.hasPendingCustomRefresh,
        setPendingCustomRefresh: runtimeStatus?.setPendingCustomRefresh,
        clearPendingCustomRefresh: runtimeStatus?.clearPendingCustomRefresh,
        hasPendingUpdate: runtimeStatus?.hasPendingUpdate,
        setPendingUpdate: runtimeStatus?.setPendingUpdate,
        clearPendingUpdate: runtimeStatus?.clearPendingUpdate,
        hasPendingComfyInfoRefresh: runtimeStatus?.hasPendingComfyInfoRefresh,
        setPendingComfyInfoRefresh: runtimeStatus?.setPendingComfyInfoRefresh,
        clearPendingComfyInfoRefresh: runtimeStatus?.clearPendingComfyInfoRefresh,
        saveComfyCheckMode: runtimeSetup?.saveComfyCheckMode,
        catalogByGroup: runtimeSetup?.catalogByGroup,
        moduleCatalogByGroup: runtimeSetup?.moduleCatalogByGroup,
        moduleCounts: runtimeSetup?.moduleCounts,
        moduleOptions: runtimeSetup?.moduleOptions,
        moduleBadges: runtimeSetup?.moduleBadges,
        moduleNodeDiffs: runtimeSetup?.moduleNodeDiffs,
        moduleInlineStatus: runtimeSetup?.moduleInlineStatus,
        updatedModulesSession: runtimeSetup?.updatedModulesSession,
        isPickerAlive: runtimeSetup?.isPickerAlive,
        fetchNodeCatalogApi: runtimeSetup?.fetchNodeCatalogApi,
        fetchModuleInfoApi: runtimeSetup?.fetchModuleInfoApi,
        fetchComfyUIInfoApi: runtimeSetup?.fetchComfyUIInfoApi,
        refreshModuleRuntimeStateApi: runtimeSetup?.refreshModuleRuntimeStateApi,
        fetchModuleRefreshStatusApi: runtimeSetup?.fetchModuleRefreshStatusApi,
        acknowledgeAllModuleNoveltyApi: runtimeSetup?.acknowledgeAllModuleNoveltyApi,
        startModuleUpdateApi: runtimeSetup?.startModuleUpdateApi,
        fetchModuleUpdateStatusApi: runtimeSetup?.fetchModuleUpdateStatusApi,
        installModuleRequirementsApi: runtimeSetup?.installModuleRequirementsApi,
        installComfyUIRequirementsApi: runtimeSetup?.installComfyUIRequirementsApi,
        debugUi: runtimeSetup?.debugUi,
        processUi: runtimeSetup?.processUi,
        setProcessTarget: runtimeSetup?.setProcessTarget,
        setModuleInlineStatus: runtimeSetup?.setModuleInlineStatus,
        disposePickerInstance: runtimeSetup?.disposePickerInstance,
    };
}
