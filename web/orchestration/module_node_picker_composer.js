/**
 * Module: web/orchestration/module_node_picker_composer.js
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

import { app } from "../../../scripts/app.js";
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
} from "../constants/module_node_picker_constants.js";
import {
    bindModuleNodesTabRelay,
    unbindModuleNodesTabRelay,
} from "../module_node_picker_tab_relay.js";
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
} from "../api/module_node_picker_api.js";
import {
    fmtDate,
    moduleBadgesFromInfo,
    moduleBadgesFromModuleEntry,
    formatModuleOption,
} from "../ui/module_node_picker_formatters.js";
import {
    formatRefreshLine,
    formatUpdateLine,
} from "../ui/module_node_picker_status.js";
import { createModuleNodePickerLayout } from "../ui/module_node_picker_layout.js";
import {
    centerNodeInCanvas,
    createNodeFromCatalogInfo,
} from "../ui/module_node_picker_node_factory.js";
import { runModuleNodePickerStartupLoad } from "./module_node_picker_bindings.js";
import { isCanceledRequestError } from "./module_node_picker_error_utils.js";
import { initializeModuleNodePickerRuntime } from "./module_node_picker_runtime_bootstrap.js";
import { createModuleNodePickerRuntimeSetup } from "./module_node_picker_runtime_setup.js";
import { createModuleNodePickerUiStage } from "./module_node_picker_ui_stage.js";
import { createModuleNodePickerFlowStage } from "./module_node_picker_flow_stage.js";

/**
 * Render Module Node Picker UI and bind all panel event handlers.
 */
export function renderModuleNodePicker(container) {
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
    const runtimeSetup = createModuleNodePickerRuntimeSetup({
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
    });
    apiClientRef = runtimeSetup.apiClient;
    const pickerStore = runtimeSetup.pickerStore;
    const diagnosticsLogger = runtimeSetup.diagnosticsLogger;
    const {
        loadCustomStatusChecked,
        saveCustomStatusChecked,
        loadComfyStatusChecked,
        saveComfyStatusChecked,
        loadComfyInfoSnapshot,
        saveComfyInfoSnapshot,
        hasPendingCustomRefresh,
        setPendingCustomRefresh,
        clearPendingCustomRefresh,
        hasPendingUpdate,
        setPendingUpdate,
        clearPendingUpdate,
        hasPendingComfyInfoRefresh,
        setPendingComfyInfoRefresh,
        clearPendingComfyInfoRefresh,
    } = runtimeSetup.runtimeStatus;
    const saveComfyCheckMode = runtimeSetup.saveComfyCheckMode;
    const catalogByGroup = runtimeSetup.catalogByGroup;
    const moduleCatalogByGroup = runtimeSetup.moduleCatalogByGroup;
    const moduleCounts = runtimeSetup.moduleCounts;
    const moduleOptions = runtimeSetup.moduleOptions;
    const moduleBadges = runtimeSetup.moduleBadges;
    const moduleNodeDiffs = runtimeSetup.moduleNodeDiffs;
    const moduleInlineStatus = runtimeSetup.moduleInlineStatus;
    const updatedModulesSession = runtimeSetup.updatedModulesSession;
    const isPickerAlive = runtimeSetup.isPickerAlive;
    const fetchNodeCatalogApi = runtimeSetup.fetchNodeCatalogApi;
    const fetchModuleInfoApi = runtimeSetup.fetchModuleInfoApi;
    const fetchComfyUIInfoApi = runtimeSetup.fetchComfyUIInfoApi;
    const refreshModuleRuntimeStateApi = runtimeSetup.refreshModuleRuntimeStateApi;
    const fetchModuleRefreshStatusApi = runtimeSetup.fetchModuleRefreshStatusApi;
    const acknowledgeAllModuleNoveltyApi = runtimeSetup.acknowledgeAllModuleNoveltyApi;
    const startModuleUpdateApi = runtimeSetup.startModuleUpdateApi;
    const fetchModuleUpdateStatusApi = runtimeSetup.fetchModuleUpdateStatusApi;
    const installModuleRequirementsApi = runtimeSetup.installModuleRequirementsApi;
    const installComfyUIRequirementsApi = runtimeSetup.installComfyUIRequirementsApi;
    const debugUi = runtimeSetup.debugUi;
    const processUi = runtimeSetup.processUi;
    const setProcessTarget = runtimeSetup.setProcessTarget;
    const setModuleInlineStatus = runtimeSetup.setModuleInlineStatus;
    const disposePickerInstance = runtimeSetup.disposePickerInstance;
    const getCurrentLogMode = () => (Boolean(pickerStore?.get?.("debugEnabled")) ? "verbose" : "summary");

    let loadModuleInfo = async () => {};
    let renderNodeList = () => {};
    let renderModuleInfo = () => {};
    let setExpandedModule = () => {};
    let loadCatalog = async () => {};

    const uiStage = createModuleNodePickerUiStage({
        shouldContinue: isPickerAlive,
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
        setExpandedModule: (value) => setExpandedModule(value),
        getRenderNodeList: () => renderNodeList,
        getLoadModuleInfo: () => loadModuleInfo,
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
    });
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
        app,
        root,
        sidebarTabId: SIDEBAR_TAB_ID,
        onDiag: (diag) => debugUi?.setDiagnosticText?.(diag),
    });

    let installComfyUIRequirementsFlow = async () => {};
    let maybeInstallChangedRequirements = async () => {};
    let runModuleUpdate = async () => {};
    let refreshComfyUIInfoFlow = async () => {};
    let refreshCustomNodesInfoFlow = async () => {};
    let refreshModuleInfoFlow = async () => {};
    let installSingleModuleRequirementsFlow = async () => {};
    let resumePendingCustomRefreshFlow = async () => {};
    let resumePendingModuleUpdateFlow = async () => {};
    let resumePendingComfyInfoRefreshFlow = async () => {};
    const flowStage = createModuleNodePickerFlowStage({
        shouldContinue: isPickerAlive,
        fetchModuleRefreshStatus: fetchModuleRefreshStatusApi,
        fetchModuleUpdateStatus: fetchModuleUpdateStatusApi,
        formatRefreshLine,
        formatUpdateLine,
        setRefreshLine,
        getProcessTarget: () => processUi.getTarget(),
        customAlert,
        customAlertText,
        getSelectedGroup,
        getSelectedModule: () => String(nodeSelect.value || ""),
        getSelectedModuleTrimmed: () => String(nodeSelect.value || "").trim(),
        fetchModuleInfo: fetchModuleInfoApi,
        fetchNodeCatalog: fetchNodeCatalogApi,
        getComfyMode: () => comfyModeSelect.value,
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
        nodeSelect,
        clearModuleInfo: () => {
            moduleInfo.innerHTML = "";
        },
        nodeList,
        setActionBusy,
        setProcessTarget,
        setProcessAction,
        setCustomRefreshCardLine,
        startModuleUpdate: startModuleUpdateApi,
        installModuleRequirements: installModuleRequirementsApi,
        installComfyUIRequirements: installComfyUIRequirementsApi,
        fetchComfyUIInfo: fetchComfyUIInfoApi,
        getLogMode: getCurrentLogMode,
        refreshModuleRuntimeState: refreshModuleRuntimeStateApi,
        acknowledgeAllModuleNovelty: acknowledgeAllModuleNoveltyApi,
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
        moduleInfo,
        updatedModulesSession,
        setHelpHintText,
        setHelpModuleCardHint,
        setHelpModuleSummary,
        createNodeByInfo: (nodeInfo) => createNodeFromCatalogInfo(nodeInfo, LiteGraph),
        app,
        centerNode: (node) => centerNodeInCanvas(node, app),
        fmtDate,
        getActionBusy: () => busyUi.getActionBusy(),
        getNodesForSelectedGroup,
        getInlineStatus: (moduleName) => moduleInlineStatus.get(moduleName) || null,
    });
    pollingController = flowStage.pollingController;
    catalogController = flowStage.catalogController;
    modulePanelController = flowStage.modulePanelController;
    loadModuleInfo = (options = {}) => flowStage.loadModuleInfo(options);
    loadCatalog = (options = {}) => flowStage.loadCatalog(options);
    renderNodeList = () => flowStage.renderNodeList();
    renderModuleInfo = (info) => flowStage.renderModuleInfo(info);
    setExpandedModule = (value) => flowStage.setExpandedModule(value);
    installComfyUIRequirementsFlow = (...args) => flowStage.installComfyUIRequirementsFlow(...args);
    maybeInstallChangedRequirements = (...args) => flowStage.maybeInstallChangedRequirements(...args);
    runModuleUpdate = (...args) => flowStage.runModuleUpdate(...args);
    refreshComfyUIInfoFlow = (...args) => flowStage.refreshComfyUIInfoFlow(...args);
    refreshCustomNodesInfoFlow = (...args) => flowStage.refreshCustomNodesInfoFlow(...args);
    refreshModuleInfoFlow = (...args) => flowStage.refreshModuleInfoFlow(...args);
    installSingleModuleRequirementsFlow = (...args) => flowStage.installSingleModuleRequirementsFlow(...args);
    resumePendingCustomRefreshFlow = (...args) => flowStage.resumePendingCustomRefreshFlow(...args);
    resumePendingModuleUpdateFlow = (...args) => flowStage.resumePendingModuleUpdateFlow(...args);
    resumePendingComfyInfoRefreshFlow = (...args) => flowStage.resumePendingComfyInfoRefreshFlow(...args);

    const runtimeBootstrap = initializeModuleNodePickerRuntime({
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
        loadModuleInfo,
        isActionBusy: () => busyUi.isActionBusy(),
        setCustomStatusChecked,
        setProcessTarget,
        runModuleUpdate,
        installComfyUIRequirementsFlow,
        refreshComfyUIInfoFlow,
        saveComfyCheckMode,
        loadCatalog,
        refreshCustomNodesInfoFlow,
        setExpandedModule: (value) => setExpandedModule(value),
        statusCards,
        hasPendingComfyInfoRefresh,
        loadComfyInfoSnapshot,
        renderComfyAlert,
        shouldContinue: isPickerAlive,
        setStartupBusy,
        startCatalogStartupLoad: (options = {}) => runModuleNodePickerStartupLoad({
            pickerStore,
            defaultModule: DEFAULT_MODULE,
            loadCatalog,
            shouldContinue: isPickerAlive,
            startupRetries: 2,
            startupRetryDelayMs: 250,
            onSettled: options?.onSettled,
        }),
        hasPendingCustomRefresh,
        hasPendingUpdate,
        resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow,
    });
    unbindPickerEvents = runtimeBootstrap.unbindPickerEvents;
    cancelStartupLoad = runtimeBootstrap.cancelStartupLoad;
}
