/**
 * Module: web/orchestration/module_node_picker_runtime_setup.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime setup for Module Node Picker base services/state.
 *
 * Purpose:
 *   Initializes runtime context, lifecycle guard, API client, debug panel, and
 *   process controller so composer code can focus on wiring higher-level flows.
 */

import { createProcessUiController } from "../ui/module_node_picker_process.js";
import { createModuleNodePickerApiClient } from "./module_node_picker_api_client.js";
import { createModuleNodePickerDebugUi } from "./module_node_picker_debug_ui.js";
import { createModuleNodePickerLifecycle } from "./module_node_picker_lifecycle.js";
import { createModuleNodePickerRuntimeContext } from "../state/module_node_picker_runtime_context.js";

/**
 * Build per-render runtime services and state holders for the picker.
 */
export function createModuleNodePickerRuntimeSetup(context = {}) {
    const runtimeContext = createModuleNodePickerRuntimeContext({
        windowObj: context?.windowObj,
        defaultModule: context?.defaultModule,
        keys: context?.runtimeKeys || {},
    });
    const pickerStore = runtimeContext.pickerStore;
    const diagnosticsLogger = runtimeContext.diagnosticsLogger;
    const runtimeStatus = runtimeContext.runtimeStatus;

    const saveComfyCheckMode = (mode) => runtimeContext.saveComfyCheckMode(mode);
    if (context?.comfyModeSelect) {
        context.comfyModeSelect.value = runtimeContext.comfyCheckMode;
    }

    const catalogByGroup = new Map();
    const moduleCatalogByGroup = new Map();
    const moduleCounts = new Map();
    const moduleOptions = new Map();
    const moduleBadges = new Map();
    const moduleNodeDiffs = new Map();
    const moduleInlineStatus = new Map();
    const updatedModulesSession = new Set();

    let debugUi = null;
    const processUi = createProcessUiController({
        processHost: context?.processHost,
        refreshLine: context?.refreshLine,
        processActions: context?.processActions,
        comfyAlert: context?.comfyAlert,
        customAlert: context?.customAlert,
        diagnosticsLogger,
        defaultTarget: () => "custom",
    });

    const lifecycle = createModuleNodePickerLifecycle({
        getCatalogController: context?.getCatalogController,
        getPollingController: context?.getPollingController,
        getUnbindPickerEvents: context?.getUnbindPickerEvents,
        getCancelStartupLoad: context?.getCancelStartupLoad,
        getDebugUi: () => debugUi,
        getProcessUi: () => processUi,
        getApiClient: context?.getApiClient,
        unbindTabRelay: context?.unbindTabRelay,
        container: context?.container,
        cleanupKey: context?.cleanupKey,
    });
    const isPickerAlive = () => lifecycle.isPickerAlive();

    const apiClient = createModuleNodePickerApiClient({
        fetchNodeCatalog: context?.fetchNodeCatalog,
        fetchModuleInfo: context?.fetchModuleInfo,
        fetchComfyUIInfo: context?.fetchComfyUIInfo,
        refreshModuleRuntimeState: context?.refreshModuleRuntimeState,
        fetchModuleRefreshStatus: context?.fetchModuleRefreshStatus,
        acknowledgeAllModuleNovelty: context?.acknowledgeAllModuleNovelty,
        startModuleUpdate: context?.startModuleUpdate,
        fetchModuleUpdateStatus: context?.fetchModuleUpdateStatus,
        installModuleRequirements: context?.installModuleRequirements,
        installComfyUIRequirements: context?.installComfyUIRequirements,
    });

    debugUi = createModuleNodePickerDebugUi({
        shouldContinue: isPickerAlive,
        windowObj: context?.windowObj,
        debugStateKey: context?.debugStateKey,
        pickerStore,
        diagnosticsLogger,
        debugToggle: context?.debugToggle,
        debugCard: context?.debugCard,
        debugCopyBtn: context?.debugCopyBtn,
        diagnostics: context?.diagnostics,
        onCopyStatus: (message) => {
            const showStatus = context?.getShowHelpStatus?.();
            if (typeof showStatus === "function") {
                showStatus(message);
            }
        },
    });

    const setProcessTarget = (target) => {
        if (!isPickerAlive()) {
            return;
        }
        processUi.setTarget(target);
    };

    const setModuleInlineStatus = (moduleName, text, tone = "neutral") => {
        if (!isPickerAlive()) {
            return;
        }
        const key = String(moduleName || "").trim();
        if (!key) {
            return;
        }
        if (!text) {
            moduleInlineStatus.delete(key);
            return;
        }
        moduleInlineStatus.set(key, {
            text: String(text),
            tone: String(tone || "neutral"),
        });
    };

    return {
        runtimeContext,
        pickerStore,
        diagnosticsLogger,
        runtimeStatus,
        saveComfyCheckMode,
        catalogByGroup,
        moduleCatalogByGroup,
        moduleCounts,
        moduleOptions,
        moduleBadges,
        moduleNodeDiffs,
        moduleInlineStatus,
        updatedModulesSession,
        isPickerAlive,
        apiClient,
        fetchNodeCatalogApi: apiClient.fetchNodeCatalogApi,
        fetchModuleInfoApi: apiClient.fetchModuleInfoApi,
        fetchComfyUIInfoApi: apiClient.fetchComfyUIInfoApi,
        refreshModuleRuntimeStateApi: apiClient.refreshModuleRuntimeStateApi,
        fetchModuleRefreshStatusApi: apiClient.fetchModuleRefreshStatusApi,
        acknowledgeAllModuleNoveltyApi: apiClient.acknowledgeAllModuleNoveltyApi,
        startModuleUpdateApi: apiClient.startModuleUpdateApi,
        fetchModuleUpdateStatusApi: apiClient.fetchModuleUpdateStatusApi,
        installModuleRequirementsApi: apiClient.installModuleRequirementsApi,
        installComfyUIRequirementsApi: apiClient.installComfyUIRequirementsApi,
        debugUi,
        processUi,
        setProcessTarget,
        setModuleInlineStatus,
        disposePickerInstance: () => lifecycle.dispose(),
    };
}
