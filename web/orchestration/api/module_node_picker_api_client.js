/**
 * Module: web/orchestration/api/module_node_picker_api_client.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Lifecycle-bound API client factory for Module Node Picker.
 *
 * Purpose:
 *   Centralizes AbortController wiring so all picker API calls share one
 *   per-render cancellation scope and are aborted on picker dispose.
 */

/**
 * Create per-render API client bound to one AbortController lifecycle.
 */
export function createModuleNodePickerApiClient(deps = {}) {
    const abortController = typeof AbortController === "function"
        ? new AbortController()
        : null;
    const apiSignal = () => abortController?.signal;

    const fetchNodeCatalog = deps.fetchNodeCatalog;
    const fetchModuleInfo = deps.fetchModuleInfo;
    const fetchComfyUIInfo = deps.fetchComfyUIInfo;
    const refreshModuleRuntimeState = deps.refreshModuleRuntimeState;
    const fetchModuleRefreshStatus = deps.fetchModuleRefreshStatus;
    const acknowledgeAllModuleNovelty = deps.acknowledgeAllModuleNovelty;
    const startModuleUpdate = deps.startModuleUpdate;
    const fetchModuleUpdateStatus = deps.fetchModuleUpdateStatus;
    const installModuleRequirements = deps.installModuleRequirements;
    const installComfyUIRequirements = deps.installComfyUIRequirements;

    return {
        fetchNodeCatalogApi: (comfyMode) =>
            fetchNodeCatalog(comfyMode, { signal: apiSignal() }),
        fetchModuleInfoApi: (group, moduleName, options = {}) =>
            fetchModuleInfo(group, moduleName, { ...(options || {}), signal: apiSignal() }),
        fetchComfyUIInfoApi: (forceRefresh = true, acknowledge = true, comfyMode = "releases", options = {}) =>
            fetchComfyUIInfo(forceRefresh, acknowledge, comfyMode, { ...(options || {}), signal: apiSignal() }),
        refreshModuleRuntimeStateApi: (options = {}) =>
            refreshModuleRuntimeState({ ...(options || {}), signal: apiSignal() }),
        fetchModuleRefreshStatusApi: (options = {}) =>
            fetchModuleRefreshStatus({ ...(options || {}), signal: apiSignal() }),
        acknowledgeAllModuleNoveltyApi: (options = {}) =>
            acknowledgeAllModuleNovelty({ ...(options || {}), signal: apiSignal() }),
        startModuleUpdateApi: (scope, moduleName, options = {}) =>
            startModuleUpdate(scope, moduleName, { ...(options || {}), signal: apiSignal() }),
        fetchModuleUpdateStatusApi: (options = {}) =>
            fetchModuleUpdateStatus({ ...(options || {}), signal: apiSignal() }),
        installModuleRequirementsApi: (modules, options = {}) =>
            installModuleRequirements(modules, { ...(options || {}), signal: apiSignal() }),
        installComfyUIRequirementsApi: (options = {}) =>
            installComfyUIRequirements({ ...(options || {}), signal: apiSignal() }),
        dispose: () => {
            try {
                abortController?.abort?.();
            } catch (_err) {
                // Ignore stale abort-controller errors.
            }
        },
    };
}
