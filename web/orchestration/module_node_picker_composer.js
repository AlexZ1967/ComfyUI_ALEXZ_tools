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
import {
    createProcessUiController,
} from "../ui/module_node_picker_process.js";
import { createModuleNodePickerLayout } from "../ui/module_node_picker_layout.js";
import {
    centerNodeInCanvas,
    createNodeFromCatalogInfo,
} from "../ui/module_node_picker_node_factory.js";
import {
    bindModuleNodePickerEvents,
    runModuleNodePickerStartupLoad,
} from "./module_node_picker_bindings.js";
import { createModuleNodePickerApiClient } from "./module_node_picker_api_client.js";
import { createModuleNodePickerFlowWiring } from "./module_node_picker_flow_wiring.js";
import { isCanceledRequestError } from "./module_node_picker_error_utils.js";
import { createModuleNodePickerDebugUi } from "./module_node_picker_debug_ui.js";
import { createModuleNodePickerUiControllers } from "./module_node_picker_ui_controllers.js";
import { runStartupCoordinator } from "./module_node_picker_startup_flow.js";
import { createModuleNodePickerLifecycle } from "./module_node_picker_lifecycle.js";
import { createModuleNodePickerRuntimeContext } from "../state/module_node_picker_runtime_context.js";

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

    const runtimeContext = createModuleNodePickerRuntimeContext({
        windowObj: window,
        defaultModule: DEFAULT_MODULE,
        keys: {
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
    });
    const pickerStore = runtimeContext.pickerStore;
    const diagnosticsLogger = runtimeContext.diagnosticsLogger;
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
    } = runtimeContext.runtimeStatus;

    comfyModeSelect.value = runtimeContext.comfyCheckMode;
    const saveComfyCheckMode = (mode) => runtimeContext.saveComfyCheckMode(mode);

    let showHelpStatus = () => {};
    let debugUi = null;

    const catalogByGroup = new Map();
    const moduleCatalogByGroup = new Map();
    const moduleCounts = new Map();
    const moduleOptions = new Map();
    const moduleBadges = new Map();
    const moduleNodeDiffs = new Map();
    const moduleInlineStatus = new Map();
    const updatedModulesSession = new Set();
    let pollingController = null;
    let customModulesNeedUpdate = 0;
    let modulePanelController = null;
    let catalogController = null;
    let selectionController = null;
    let statusCards = null;
    let unbindPickerEvents = () => {};
    let processUi = null;
    let cancelStartupLoad = () => {};
    let apiClient = null;
    const lifecycle = createModuleNodePickerLifecycle({
        getCatalogController: () => catalogController,
        getPollingController: () => pollingController,
        getUnbindPickerEvents: () => unbindPickerEvents,
        getCancelStartupLoad: () => cancelStartupLoad,
        getDebugUi: () => debugUi,
        getProcessUi: () => processUi,
        getApiClient: () => apiClient,
        unbindTabRelay: () => unbindModuleNodesTabRelay(),
        container,
        cleanupKey: PICKER_CLEANUP_KEY,
    });
    // Keep async/UI flows active for this picker instance even if the root is
    // temporarily detached during sidebar transitions; lifecycle is governed by
    // explicit dispose, not transient DOM attachment state.
    const isPickerAlive = () => lifecycle.isPickerAlive();
    apiClient = createModuleNodePickerApiClient({
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
    const fetchNodeCatalogApi = apiClient.fetchNodeCatalogApi;
    const fetchModuleInfoApi = apiClient.fetchModuleInfoApi;
    const fetchComfyUIInfoApi = apiClient.fetchComfyUIInfoApi;
    const refreshModuleRuntimeStateApi = apiClient.refreshModuleRuntimeStateApi;
    const fetchModuleRefreshStatusApi = apiClient.fetchModuleRefreshStatusApi;
    const acknowledgeAllModuleNoveltyApi = apiClient.acknowledgeAllModuleNoveltyApi;
    const startModuleUpdateApi = apiClient.startModuleUpdateApi;
    const fetchModuleUpdateStatusApi = apiClient.fetchModuleUpdateStatusApi;
    const installModuleRequirementsApi = apiClient.installModuleRequirementsApi;
    const installComfyUIRequirementsApi = apiClient.installComfyUIRequirementsApi;
    debugUi = createModuleNodePickerDebugUi({
        shouldContinue: isPickerAlive,
        windowObj: window,
        debugStateKey: NODE_PICKER_DEBUG_KEY,
        pickerStore,
        diagnosticsLogger,
        debugToggle,
        debugCard,
        debugCopyBtn,
        diagnostics,
        onCopyStatus: (message) => showHelpStatus(message),
    });
    const disposePickerInstance = () => lifecycle.dispose();

    /**
     * Store one-line module action result shown inside module card.
     */
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

    processUi = createProcessUiController({
        processHost,
        refreshLine,
        processActions,
        comfyAlert,
        customAlert,
        diagnosticsLogger,
        defaultTarget: () => "custom",
    });

    /**
     * Mount progress inline block into the selected top status card.
     */
    const setProcessTarget = (target) => {
        if (!isPickerAlive()) {
            return;
        }
        processUi.setTarget(target);
    };

    let loadModuleInfo = async () => {};
    let renderNodeList = () => {};
    let renderModuleInfo = () => {};
    let setExpandedModule = () => {};
    let loadCatalog = async () => {};
    const uiControllers = createModuleNodePickerUiControllers({
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
    selectionController = uiControllers.selectionController;
    const isCustomCategory = uiControllers.isCustomCategory;
    const getSelectedGroup = uiControllers.getSelectedGroup;
    const syncPickerSelectionState = uiControllers.syncPickerSelectionState;
    const getNodesForSelectedGroup = uiControllers.getNodesForSelectedGroup;
    const fillModuleSelect = uiControllers.fillModuleSelect;
    const fillGroupSelect = uiControllers.fillGroupSelect;
    const busyUi = uiControllers.busyUi;
    const syncBusyUiState = uiControllers.syncBusyUiState;
    const setCatalogControlsLoading = uiControllers.setCatalogControlsLoading;
    const setActionBusy = uiControllers.setActionBusy;
    const setStartupBusy = uiControllers.setStartupBusy;
    const {
        setProcessAction,
        setRefreshLine,
        setCustomRefreshCardLine,
        setHelpText,
        setHelpHintText,
        setHelpModuleSummary,
        setHelpModuleCardHint,
    } = uiControllers.viewHelpers;
    showHelpStatus = setHelpText;
    statusCards = uiControllers.statusCards;
    const renderComfyAlert = uiControllers.renderComfyAlert;
    const renderCustomAlert = uiControllers.renderCustomAlert;
    const syncUpdateAllButton = uiControllers.syncUpdateAllButton;
    const setCustomStatusChecked = uiControllers.setCustomStatusChecked;
    const setComfyStatusChecked = uiControllers.setComfyStatusChecked;

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
    const flowWiring = createModuleNodePickerFlowWiring({
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
    pollingController = flowWiring.pollingController;
    catalogController = flowWiring.catalogController;
    modulePanelController = flowWiring.modulePanelController;
    loadModuleInfo = (options = {}) => flowWiring.loadModuleInfo(options);
    loadCatalog = (options = {}) => flowWiring.loadCatalog(options);
    renderNodeList = () => flowWiring.renderNodeList();
    renderModuleInfo = (info) => flowWiring.renderModuleInfo(info);
    setExpandedModule = (value) => flowWiring.setExpandedModule(value);
    installComfyUIRequirementsFlow = (...args) => flowWiring.actionFlows.installComfyUIRequirementsFlow(...args);
    maybeInstallChangedRequirements = (...args) => flowWiring.actionFlows.maybeInstallChangedRequirements(...args);
    runModuleUpdate = (...args) => flowWiring.actionFlows.runModuleUpdate(...args);
    refreshComfyUIInfoFlow = (...args) => flowWiring.actionFlows.refreshComfyUIInfoFlow(...args);
    refreshCustomNodesInfoFlow = (...args) => flowWiring.actionFlows.refreshCustomNodesInfoFlow(...args);
    refreshModuleInfoFlow = (...args) => flowWiring.actionFlows.refreshModuleInfoFlow(...args);
    installSingleModuleRequirementsFlow = (...args) => flowWiring.actionFlows.installSingleModuleRequirementsFlow(...args);
    resumePendingCustomRefreshFlow = (...args) => flowWiring.actionFlows.resumePendingCustomRefreshFlow(...args);
    resumePendingModuleUpdateFlow = (...args) => flowWiring.actionFlows.resumePendingModuleUpdateFlow(...args);
    resumePendingComfyInfoRefreshFlow = (...args) => flowWiring.actionFlows.resumePendingComfyInfoRefreshFlow(...args);

    unbindPickerEvents = bindModuleNodePickerEvents({
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
    }) || (() => {});

    // Restore last ComfyUI status card across widget switches in current session.
    if (statusCards?.getComfyStatusChecked?.() && !hasPendingComfyInfoRefresh()) {
        const lastComfyInfo = loadComfyInfoSnapshot();
        if (lastComfyInfo) {
            renderComfyAlert(lastComfyInfo);
        }
    }

    cancelStartupLoad = runStartupCoordinator({
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
        hasPendingComfyInfoRefresh,
        resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow,
    });
}
