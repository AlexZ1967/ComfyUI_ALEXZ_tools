/**
 * Module: web/orchestration/core/composition/module_node_picker_composer.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module Node Picker composition entry.
 *
 * Purpose:
 *   Composes picker layout, state/controllers, and async flows for module
 *   catalog rendering and update operations.
 */

import {
    SIDEBAR_TAB_ID,
    DEFAULT_MODULE,
    NODE_PICKER_DEBUG_KEY,
    NODE_PICKER_DEBUG_STORAGE_KEY,
    NODE_PICKER_SELECTED_GROUP_STORAGE_KEY,
    NODE_PICKER_SELECTED_MODULE_STORAGE_KEY,
    COMFYUI_CHECK_MODE_STORAGE_KEY,
    MODULE_PICKER_RUNTIME_STATE_KEY,
    LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY,
    LEGACY_PENDING_CUSTOM_REFRESH_STORAGE_KEY,
    LEGACY_PENDING_UPDATE_STORAGE_KEY,
    PICKER_CLEANUP_KEY,
    GROUP_LABELS,
    COMFY_GROUP_ORDER,
    MODULE_MARK_UPDATED,
    MODULE_MARK_REMOTE_UPDATE,
} from "../../../constants/module_node_picker_constants.js";
import {
    bindModuleNodesTabRelay,
    unbindModuleNodesTabRelay,
} from "../../../module_node_picker_tab_relay.js";
import {
    fetchNodeCatalog,
    fetchModuleInfo,
    fetchComfyUIInfo,
    refreshModuleRuntimeState,
    fetchModuleRefreshStatus,
    acknowledgeAllModuleNovelty,
    startModuleUpdate,
    fetchModuleUpdateStatus,
    installModuleRequirements,
    installComfyUIRequirements,
} from "../../../api/module_node_picker_api.js";
import {
    fmtDate,
    moduleBadgesFromInfo,
    moduleBadgesFromModuleEntry,
    formatModuleOption,
} from "../../../ui/module_node_picker_formatters.js";
import {
    formatRefreshLine,
    formatUpdateLine,
} from "../../../ui/module_node_picker_status.js";
import { createModuleNodePickerLayout } from "../../../ui/module_node_picker_layout.js";
import {
    centerNodeInCanvas,
    createNodeFromCatalogInfo,
} from "../../../ui/module_node_picker_node_factory.js";
import { isCanceledRequestError } from "../infra/module_node_picker_error_utils.js";
import { initializeModuleNodePickerRuntime } from "../../runtime/bootstrap/module_node_picker_runtime_bootstrap.js";
import { createModuleNodePickerRuntimeSetup } from "../../runtime/bootstrap/module_node_picker_runtime_setup.js";
import { createModuleNodePickerUiStage } from "../../ui/module_node_picker_ui_stage.js";
import { createModuleNodePickerFlowStage } from "../../flow/stage/module_node_picker_flow_stage.js";
import { createModuleNodePickerStageBridge } from "./module_node_picker_stage_bridge.js";
import { createModuleNodePickerRuntimeBootstrapBindings } from "../../runtime/bootstrap/module_node_picker_runtime_bootstrap_bindings.js";
import { projectModuleNodePickerRuntimeSetup } from "../../runtime/bootstrap/module_node_picker_runtime_projection.js";
import {
    buildFlowStageContext,
    buildRuntimeSetupContext,
    buildRuntimeBootstrapContext,
    buildUiStageContext,
} from "./module_node_picker_context_builders.js";

/**
 * Render Module Node Picker UI and bind all panel event handlers.
 */
export function renderModuleNodePicker(container, options = {}) {
    const appInstance = options?.appInstance;
    if (!appInstance) {
        if (container) {
            container.innerHTML = "<div style=\"padding:8px;color:#f66;\">Module Nodes initialization error: app instance is missing.</div>";
        }
        return;
    }
    const prevCleanup = container?.[PICKER_CLEANUP_KEY];
    if (typeof prevCleanup === "function") {
        try {
            prevCleanup();
        } catch (_err) {
            // Ignore stale cleanup errors from previous picker instance.
        }
    }
    unbindModuleNodesTabRelay();

    const {
        root,
        debugToggle,
        debugCard,
        debugCopyBtn,
        diagnostics,
        warmupHint,
        comfyInfoBtn,
        refreshBtn,
        comfyModeSelect,
        comfyAlert,
        comfyAlertText,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        customAlert,
        customAlertText,
        updateAllBtn,
        processHost,
        categorySelect,
        groupSelect,
        nodeSelect,
        moduleFilter,
        refreshLine,
        processActions,
        help,
        moduleInfo,
        nodeList,
    } = createModuleNodePickerLayout(container);

    let showHelpStatus = () => {};
    let pollingController = null;
    let customModulesNeedUpdate = 0;
    let modulePanelController = null;
    let catalogController = null;
    let selectionController = null;
    let statusCards = null;
    let unbindPickerEvents = () => {};
    let cancelStartupLoad = () => {};
    let apiClientRef = null;
    const runtimeSetup = createModuleNodePickerRuntimeSetup(buildRuntimeSetupContext({
        windowObj: window,
        defaultModule: DEFAULT_MODULE,
        runtimeKeys: {
            debugRuntimeKey: NODE_PICKER_DEBUG_KEY,
            selectedGroupStorageKey: NODE_PICKER_SELECTED_GROUP_STORAGE_KEY,
            selectedModuleStorageKey: NODE_PICKER_SELECTED_MODULE_STORAGE_KEY,
            debugStorageKey: NODE_PICKER_DEBUG_STORAGE_KEY,
            runtimeStateKey: MODULE_PICKER_RUNTIME_STATE_KEY,
            legacyCustomStatusCheckedKey: LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY,
            legacyPendingCustomRefreshKey: LEGACY_PENDING_CUSTOM_REFRESH_STORAGE_KEY,
            legacyPendingUpdateKey: LEGACY_PENDING_UPDATE_STORAGE_KEY,
            comfyCheckModeStorageKey: COMFYUI_CHECK_MODE_STORAGE_KEY,
        },
        comfyModeSelect,
        processHost,
        refreshLine,
        processActions,
        comfyAlert,
        customAlert,
        debugToggle,
        debugCard,
        debugCopyBtn,
        diagnostics,
        debugStateKey: NODE_PICKER_DEBUG_KEY,
        getShowHelpStatus: () => showHelpStatus,
        getCatalogController: () => catalogController,
        getPollingController: () => pollingController,
        getUnbindPickerEvents: () => unbindPickerEvents,
        getCancelStartupLoad: () => cancelStartupLoad,
        getApiClient: () => apiClientRef,
        unbindTabRelay: () => unbindModuleNodesTabRelay(),
        container,
        cleanupKey: PICKER_CLEANUP_KEY,
        fetchNodeCatalog,
        fetchModuleInfo,
        fetchComfyUIInfo,
        refreshModuleRuntimeState,
        fetchModuleRefreshStatus,
        acknowledgeAllModuleNovelty,
        startModuleUpdate,
        fetchModuleUpdateStatus,
        installModuleRequirements,
        installComfyUIRequirements,
    }));
    apiClientRef = runtimeSetup.apiClient;
    const runtimeProjection = projectModuleNodePickerRuntimeSetup(runtimeSetup);
    const pickerStore = runtimeProjection.pickerStore;
    const diagnosticsLogger = runtimeProjection.diagnosticsLogger;
    const loadCustomStatusChecked = runtimeProjection.loadCustomStatusChecked;
    const saveCustomStatusChecked = runtimeProjection.saveCustomStatusChecked;
    const loadComfyStatusChecked = runtimeProjection.loadComfyStatusChecked;
    const saveComfyStatusChecked = runtimeProjection.saveComfyStatusChecked;
    const loadComfyInfoSnapshot = runtimeProjection.loadComfyInfoSnapshot;
    const saveComfyInfoSnapshot = runtimeProjection.saveComfyInfoSnapshot;
    const hasPendingCustomRefresh = runtimeProjection.hasPendingCustomRefresh;
    const setPendingCustomRefresh = runtimeProjection.setPendingCustomRefresh;
    const clearPendingCustomRefresh = runtimeProjection.clearPendingCustomRefresh;
    const hasPendingUpdate = runtimeProjection.hasPendingUpdate;
    const setPendingUpdate = runtimeProjection.setPendingUpdate;
    const clearPendingUpdate = runtimeProjection.clearPendingUpdate;
    const hasPendingComfyInfoRefresh = runtimeProjection.hasPendingComfyInfoRefresh;
    const setPendingComfyInfoRefresh = runtimeProjection.setPendingComfyInfoRefresh;
    const clearPendingComfyInfoRefresh = runtimeProjection.clearPendingComfyInfoRefresh;
    const saveComfyCheckMode = runtimeProjection.saveComfyCheckMode;
    const catalogByGroup = runtimeProjection.catalogByGroup;
    const moduleCatalogByGroup = runtimeProjection.moduleCatalogByGroup;
    const moduleCounts = runtimeProjection.moduleCounts;
    const moduleOptions = runtimeProjection.moduleOptions;
    const moduleBadges = runtimeProjection.moduleBadges;
    const moduleNodeDiffs = runtimeProjection.moduleNodeDiffs;
    const moduleInlineStatus = runtimeProjection.moduleInlineStatus;
    const updatedModulesSession = runtimeProjection.updatedModulesSession;
    const isPickerAlive = runtimeProjection.isPickerAlive;
    const fetchNodeCatalogApi = runtimeProjection.fetchNodeCatalogApi;
    const fetchModuleInfoApi = runtimeProjection.fetchModuleInfoApi;
    const fetchComfyUIInfoApi = runtimeProjection.fetchComfyUIInfoApi;
    const refreshModuleRuntimeStateApi = runtimeProjection.refreshModuleRuntimeStateApi;
    const fetchModuleRefreshStatusApi = runtimeProjection.fetchModuleRefreshStatusApi;
    const acknowledgeAllModuleNoveltyApi = runtimeProjection.acknowledgeAllModuleNoveltyApi;
    const startModuleUpdateApi = runtimeProjection.startModuleUpdateApi;
    const fetchModuleUpdateStatusApi = runtimeProjection.fetchModuleUpdateStatusApi;
    const installModuleRequirementsApi = runtimeProjection.installModuleRequirementsApi;
    const installComfyUIRequirementsApi = runtimeProjection.installComfyUIRequirementsApi;
    const debugUi = runtimeProjection.debugUi;
    const processUi = runtimeProjection.processUi;
    const setProcessTarget = runtimeProjection.setProcessTarget;
    const setModuleInlineStatus = runtimeProjection.setModuleInlineStatus;
    const disposePickerInstance = runtimeProjection.disposePickerInstance;
    const getCurrentLogMode = () => (Boolean(pickerStore?.get?.("debugEnabled")) ? "verbose" : "summary");
    const setWarmupIndicator = (running) => {
        if (!warmupHint) {
            return;
        }
        warmupHint.style.display = running ? "inline" : "none";
    };

    const stageBridge = createModuleNodePickerStageBridge();
    const deferredStage = stageBridge.deferredStage;
    const stageAdapters = stageBridge.adapters;

    const uiStage = createModuleNodePickerUiStage(buildUiStageContext({
        isPickerAlive,
        categorySelect,
        groupSelect,
        nodeSelect,
        moduleFilter,
        moduleInfo,
        nodeList,
        pickerStore,
        catalogByGroup,
        moduleCatalogByGroup,
        moduleCounts,
        moduleOptions,
        moduleBadges,
        moduleNodeDiffs,
        moduleBadgesFromModuleEntry,
        formatModuleOption,
        marks: {
            updatedMark: MODULE_MARK_UPDATED,
            remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
        },
        defaultModule: DEFAULT_MODULE,
        comfyGroupOrder: COMFY_GROUP_ORDER,
        groupLabels: GROUP_LABELS,
        setExpandedModule: (value) => stageAdapters.setExpandedModule(value),
        getRenderNodeList: () => deferredStage.renderNodeList,
        getLoadModuleInfo: () => deferredStage.loadModuleInfo,
        controls: {
            refreshBtn,
            comfyInfoBtn,
            comfyModeSelect,
            categorySelect,
            groupSelect,
            nodeSelect,
            moduleFilter,
            updateAllBtn,
            comfyUpdateBtn,
            comfyInstallReqBtn,
            moduleInfo,
            nodeList,
        },
        getProcessUi: () => processUi,
        processUi,
        customAlert,
        customAlertText,
        help,
        getComfyMode: () => comfyModeSelect.value,
        fmtDate,
        comfyAlert,
        comfyAlertText,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        updateAllBtn,
        getCustomModulesNeedUpdate: () => customModulesNeedUpdate,
        saveCustomStatusChecked,
        saveComfyStatusChecked,
        saveComfyInfoSnapshot,
        initialCustomStatusChecked: loadCustomStatusChecked(),
        initialComfyStatusChecked: loadComfyStatusChecked(),
    }));
    selectionController = uiStage.selectionController;
    const isCustomCategory = uiStage.isCustomCategory;
    const getSelectedGroup = uiStage.getSelectedGroup;
    const syncPickerSelectionState = uiStage.syncPickerSelectionState;
    const getNodesForSelectedGroup = uiStage.getNodesForSelectedGroup;
    const fillModuleSelect = uiStage.fillModuleSelect;
    const busyUi = uiStage.busyUi;
    const setCatalogControlsLoading = uiStage.setCatalogControlsLoading;
    const setActionBusy = uiStage.setActionBusy;
    const setStartupBusy = uiStage.setStartupBusy;
    const {
        setProcessAction,
        setRefreshLine,
        setCustomRefreshCardLine,
        setHelpText,
        setHelpHintText,
        setHelpModuleSummary,
        setHelpModuleCardHint,
    } = uiStage.viewHelpers;
    showHelpStatus = uiStage.showHelpStatus;
    statusCards = uiStage.statusCards;
    const renderComfyAlert = uiStage.renderComfyAlert;
    const renderCustomAlert = uiStage.renderCustomAlert;
    const syncUpdateAllButton = uiStage.syncUpdateAllButton;
    const setCustomStatusChecked = uiStage.setCustomStatusChecked;
    const setComfyStatusChecked = uiStage.setComfyStatusChecked;

    bindModuleNodesTabRelay({
        app: appInstance,
        root,
        sidebarTabId: SIDEBAR_TAB_ID,
        onDiag: (diag) => debugUi?.setDiagnosticText?.(diag),
    });

    const flowStage = createModuleNodePickerFlowStage(buildFlowStageContext({
        isPickerAlive,
        fetchModuleRefreshStatusApi,
        fetchModuleUpdateStatusApi,
        formatRefreshLine,
        formatUpdateLine,
        setRefreshLine,
        processUi,
        customAlert,
        customAlertText,
        getSelectedGroup,
        nodeSelect,
        fetchModuleInfoApi,
        fetchNodeCatalogApi,
        comfyModeSelect,
        catalogByGroup,
        moduleCounts,
        moduleOptions,
        moduleBadges,
        moduleNodeDiffs,
        formatModuleOption,
        marks: {
            updatedMark: MODULE_MARK_UPDATED,
            remoteUpdateMark: MODULE_MARK_REMOTE_UPDATE,
        },
        moduleBadgesFromInfo,
        setCatalogControlsLoading,
        setCustomModulesNeedUpdate: (value) => {
            customModulesNeedUpdate = Number(value || 0);
        },
        setWarmupIndicator,
        renderComfyAlert,
        selectionController,
        groupLabels: GROUP_LABELS,
        setHelpText,
        syncUpdateAllButton,
        comfyAlert,
        comfyAlertText,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        groupSelect,
        moduleInfo,
        nodeList,
        setActionBusy,
        setProcessTarget,
        setProcessAction,
        setCustomRefreshCardLine,
        startModuleUpdateApi,
        installModuleRequirementsApi,
        installComfyUIRequirementsApi,
        fetchComfyUIInfoApi,
        getCurrentLogMode,
        refreshModuleRuntimeStateApi,
        acknowledgeAllModuleNoveltyApi,
        setModuleInlineStatus,
        setCustomStatusChecked,
        setComfyStatusChecked,
        setPendingUpdate,
        clearPendingUpdate,
        setPendingCustomRefresh,
        clearPendingCustomRefresh,
        setPendingComfyInfoRefresh,
        clearPendingComfyInfoRefresh,
        hasPendingCustomRefresh,
        hasPendingUpdate,
        hasPendingComfyInfoRefresh,
        onMarkUpdatedModule: (mod) => updatedModulesSession.add(mod),
        isModuleMarkedUpdated: (mod) => updatedModulesSession.has(String(mod || "").trim()),
        isCanceledRequestError,
        updatedModulesSession,
        setHelpHintText,
        setHelpModuleCardHint,
        setHelpModuleSummary,
        createNodeByInfo: (nodeInfo) => createNodeFromCatalogInfo(nodeInfo, LiteGraph),
        app: appInstance,
        centerNode: (node) => centerNodeInCanvas(node, appInstance),
        fmtDate,
        busyUi,
        getNodesForSelectedGroup,
        getInlineStatus: (moduleName) => moduleInlineStatus.get(moduleName) || null,
    }));
    pollingController = flowStage.pollingController;
    catalogController = flowStage.catalogController;
    modulePanelController = flowStage.modulePanelController;
    stageBridge.wireFlowStage(flowStage);
    const runtimeBootstrapBindings = createModuleNodePickerRuntimeBootstrapBindings({
        flowStage,
        stageAdapters,
        isPickerAlive,
        pickerStore,
        defaultModule: DEFAULT_MODULE,
    });

    const runtimeBootstrap = initializeModuleNodePickerRuntime(buildRuntimeBootstrapContext({
        groupSelect,
        categorySelect,
        moduleFilter,
        nodeSelect,
        nodeList,
        updateAllBtn,
        comfyUpdateBtn,
        comfyInstallReqBtn,
        comfyInfoBtn,
        comfyModeSelect,
        refreshBtn,
        isCustomCategory,
        pickerStore,
        getSelectedGroup,
        fillModuleSelect,
        syncUpdateAllButton,
        syncPickerSelectionState,
        loadModuleInfo: runtimeBootstrapBindings.loadModuleInfo,
        busyUi,
        setCustomStatusChecked,
        setProcessTarget,
        runModuleUpdate: runtimeBootstrapBindings.runModuleUpdate,
        installComfyUIRequirementsFlow: runtimeBootstrapBindings.installComfyUIRequirementsFlow,
        refreshComfyUIInfoFlow: runtimeBootstrapBindings.refreshComfyUIInfoFlow,
        saveComfyCheckMode,
        loadCatalog: runtimeBootstrapBindings.loadCatalog,
        refreshCustomNodesInfoFlow: runtimeBootstrapBindings.refreshCustomNodesInfoFlow,
        setExpandedModule: runtimeBootstrapBindings.setExpandedModule,
        statusCards,
        hasPendingComfyInfoRefresh,
        loadComfyInfoSnapshot,
        renderComfyAlert,
        isPickerAlive,
        setStartupBusy,
        startCatalogStartupLoad: runtimeBootstrapBindings.startCatalogStartupLoad,
        hasPendingCustomRefresh,
        hasPendingUpdate,
        resumePendingCustomRefreshFlow: runtimeBootstrapBindings.resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow: runtimeBootstrapBindings.resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow: runtimeBootstrapBindings.resumePendingComfyInfoRefreshFlow,
    }));
    unbindPickerEvents = runtimeBootstrap.unbindPickerEvents;
    cancelStartupLoad = runtimeBootstrap.cancelStartupLoad;
}
