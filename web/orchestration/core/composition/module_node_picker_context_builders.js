/**
 * Module: web/orchestration/core/composition/module_node_picker_context_builders.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Context-builder helpers for Module Node Picker orchestration stages.
 *
 * Purpose:
 *   Keeps large dependency maps out of composer and builds normalized input
 *   objects for flow stage and runtime bootstrap modules without changing
 *   runtime behavior.
 */

/**
 * Build normalized dependency context for flow-stage factory.
 */
export function buildFlowStageContext(context = {}) {
    return {
        shouldContinue: context?.isPickerAlive,
        fetchModuleRefreshStatus: context?.fetchModuleRefreshStatusApi,
        fetchModuleUpdateStatus: context?.fetchModuleUpdateStatusApi,
        formatRefreshLine: context?.formatRefreshLine,
        formatUpdateLine: context?.formatUpdateLine,
        setRefreshLine: context?.setRefreshLine,
        getProcessTarget: () => context?.processUi?.getTarget?.(),
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        getSelectedGroup: context?.getSelectedGroup,
        getSelectedModule: () => String(context?.nodeSelect?.value || ""),
        getSelectedModuleTrimmed: () => String(context?.nodeSelect?.value || "").trim(),
        fetchModuleInfo: context?.fetchModuleInfoApi,
        fetchNodeCatalog: context?.fetchNodeCatalogApi,
        getComfyMode: () => context?.comfyModeSelect?.value,
        catalogByGroup: context?.catalogByGroup,
        moduleCounts: context?.moduleCounts,
        moduleOptions: context?.moduleOptions,
        moduleBadges: context?.moduleBadges,
        moduleNodeDiffs: context?.moduleNodeDiffs,
        formatModuleOption: context?.formatModuleOption,
        marks: context?.marks,
        moduleBadgesFromInfo: context?.moduleBadgesFromInfo,
        setCatalogControlsLoading: context?.setCatalogControlsLoading,
        setCustomModulesNeedUpdate: context?.setCustomModulesNeedUpdate,
        setCustomModulesUnknownUpdate: context?.setCustomModulesUnknownUpdate,
        setWarmupIndicator: context?.setWarmupIndicator,
        renderComfyAlert: context?.renderComfyAlert,
        selectionController: context?.selectionController,
        groupLabels: context?.groupLabels,
        setHelpText: context?.setHelpText,
        syncUpdateAllButton: context?.syncUpdateAllButton,
        comfyAlert: context?.comfyAlert,
        comfyAlertText: context?.comfyAlertText,
        comfyUpdateBtn: context?.comfyUpdateBtn,
        comfyInstallReqBtn: context?.comfyInstallReqBtn,
        groupSelect: context?.groupSelect,
        nodeSelect: context?.nodeSelect,
        clearModuleInfo: () => {
            if (context?.moduleInfo) {
                context.moduleInfo.innerHTML = "";
            }
        },
        nodeList: context?.nodeList,
        setActionBusy: context?.setActionBusy,
        setProcessTarget: context?.setProcessTarget,
        setProcessAction: context?.setProcessAction,
        setCustomRefreshCardLine: context?.setCustomRefreshCardLine,
        startModuleUpdate: context?.startModuleUpdateApi,
        installModuleRequirements: context?.installModuleRequirementsApi,
        installComfyUIRequirements: context?.installComfyUIRequirementsApi,
        fetchComfyUIInfo: context?.fetchComfyUIInfoApi,
        getLogMode: context?.getCurrentLogMode,
        refreshModuleRuntimeState: context?.refreshModuleRuntimeStateApi,
        acknowledgeAllModuleNovelty: context?.acknowledgeAllModuleNoveltyApi,
        setModuleInlineStatus: context?.setModuleInlineStatus,
        setCustomStatusChecked: context?.setCustomStatusChecked,
        setComfyStatusChecked: context?.setComfyStatusChecked,
        setPendingUpdate: context?.setPendingUpdate,
        clearPendingUpdate: context?.clearPendingUpdate,
        setPendingCustomRefresh: context?.setPendingCustomRefresh,
        clearPendingCustomRefresh: context?.clearPendingCustomRefresh,
        setPendingComfyInfoRefresh: context?.setPendingComfyInfoRefresh,
        clearPendingComfyInfoRefresh: context?.clearPendingComfyInfoRefresh,
        hasPendingCustomRefresh: context?.hasPendingCustomRefresh,
        hasPendingUpdate: context?.hasPendingUpdate,
        hasPendingComfyInfoRefresh: context?.hasPendingComfyInfoRefresh,
        onMarkUpdatedModule: context?.onMarkUpdatedModule,
        isModuleMarkedUpdated: context?.isModuleMarkedUpdated,
        isCanceledRequestError: context?.isCanceledRequestError,
        moduleInfo: context?.moduleInfo,
        updatedModulesSession: context?.updatedModulesSession,
        setHelpHintText: context?.setHelpHintText,
        setHelpModuleCardHint: context?.setHelpModuleCardHint,
        setHelpModuleSummary: context?.setHelpModuleSummary,
        createNodeByInfo: (nodeInfo) => context?.createNodeByInfo?.(nodeInfo),
        app: context?.app,
        centerNode: (node) => context?.centerNode?.(node),
        fmtDate: context?.fmtDate,
        getActionBusy: () => context?.busyUi?.getActionBusy?.(),
        getNodesForSelectedGroup: context?.getNodesForSelectedGroup,
        getInlineStatus: context?.getInlineStatus,
    };
}

/**
 * Build normalized dependency context for runtime-setup factory.
 */
export function buildRuntimeSetupContext(context = {}) {
    return {
        windowObj: context?.windowObj,
        defaultModule: context?.defaultModule,
        runtimeKeys: context?.runtimeKeys,
        comfyModeSelect: context?.comfyModeSelect,
        processHost: context?.processHost,
        refreshLine: context?.refreshLine,
        processActions: context?.processActions,
        comfyAlert: context?.comfyAlert,
        customAlert: context?.customAlert,
        debugToggle: context?.debugToggle,
        debugCard: context?.debugCard,
        debugCopyBtn: context?.debugCopyBtn,
        diagnostics: context?.diagnostics,
        debugStateKey: context?.debugStateKey,
        getShowHelpStatus: context?.getShowHelpStatus,
        getCatalogController: context?.getCatalogController,
        getPollingController: context?.getPollingController,
        getUnbindPickerEvents: context?.getUnbindPickerEvents,
        getCancelStartupLoad: context?.getCancelStartupLoad,
        getApiClient: context?.getApiClient,
        unbindTabRelay: context?.unbindTabRelay,
        container: context?.container,
        cleanupKey: context?.cleanupKey,
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
    };
}

/**
 * Build normalized dependency context for UI-stage factory.
 */
export function buildUiStageContext(context = {}) {
    return {
        shouldContinue: context?.isPickerAlive,
        categorySelect: context?.categorySelect,
        groupSelect: context?.groupSelect,
        nodeSelect: context?.nodeSelect,
        moduleFilter: context?.moduleFilter,
        moduleInfo: context?.moduleInfo,
        nodeList: context?.nodeList,
        pickerStore: context?.pickerStore,
        catalogByGroup: context?.catalogByGroup,
        moduleCatalogByGroup: context?.moduleCatalogByGroup,
        moduleCounts: context?.moduleCounts,
        moduleOptions: context?.moduleOptions,
        moduleBadges: context?.moduleBadges,
        moduleNodeDiffs: context?.moduleNodeDiffs,
        moduleBadgesFromModuleEntry: context?.moduleBadgesFromModuleEntry,
        formatModuleOption: context?.formatModuleOption,
        marks: context?.marks,
        defaultModule: context?.defaultModule,
        comfyGroupOrder: context?.comfyGroupOrder,
        groupLabels: context?.groupLabels,
        setExpandedModule: context?.setExpandedModule,
        getRenderNodeList: context?.getRenderNodeList,
        getLoadModuleInfo: context?.getLoadModuleInfo,
        controls: context?.controls,
        getProcessUi: context?.getProcessUi,
        processUi: context?.processUi,
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        selectionHelp: context?.selectionHelp,
        moduleHelp: context?.moduleHelp,
        getComfyMode: context?.getComfyMode,
        fmtDate: context?.fmtDate,
        comfyAlert: context?.comfyAlert,
        comfyAlertText: context?.comfyAlertText,
        comfyUpdateBtn: context?.comfyUpdateBtn,
        comfyInstallReqBtn: context?.comfyInstallReqBtn,
        updateAllBtn: context?.updateAllBtn,
        getCustomModulesNeedUpdate: context?.getCustomModulesNeedUpdate,
        getCustomModulesUnknownUpdate: context?.getCustomModulesUnknownUpdate,
        saveCustomStatusChecked: context?.saveCustomStatusChecked,
        saveComfyStatusChecked: context?.saveComfyStatusChecked,
        saveComfyInfoSnapshot: context?.saveComfyInfoSnapshot,
        initialCustomStatusChecked: context?.initialCustomStatusChecked,
        initialComfyStatusChecked: context?.initialComfyStatusChecked,
    };
}

/**
 * Build normalized dependency context for runtime-bootstrap initializer.
 */
export function buildRuntimeBootstrapContext(context = {}) {
    return {
        groupSelect: context?.groupSelect,
        categorySelect: context?.categorySelect,
        moduleFilter: context?.moduleFilter,
        nodeSelect: context?.nodeSelect,
        nodeList: context?.nodeList,
        updateAllBtn: context?.updateAllBtn,
        comfyUpdateBtn: context?.comfyUpdateBtn,
        comfyInstallReqBtn: context?.comfyInstallReqBtn,
        comfyInfoBtn: context?.comfyInfoBtn,
        comfyModeSelect: context?.comfyModeSelect,
        refreshBtn: context?.refreshBtn,
        isCustomCategory: context?.isCustomCategory,
        pickerStore: context?.pickerStore,
        getSelectedGroup: context?.getSelectedGroup,
        fillModuleSelect: context?.fillModuleSelect,
        syncUpdateAllButton: context?.syncUpdateAllButton,
        syncPickerSelectionState: context?.syncPickerSelectionState,
        loadModuleInfo: context?.loadModuleInfo,
        isActionBusy: () => context?.busyUi?.isActionBusy?.(),
        setCustomStatusChecked: context?.setCustomStatusChecked,
        setProcessTarget: context?.setProcessTarget,
        runModuleUpdate: context?.runModuleUpdate,
        installComfyUIRequirementsFlow: context?.installComfyUIRequirementsFlow,
        refreshComfyUIInfoFlow: context?.refreshComfyUIInfoFlow,
        saveComfyCheckMode: context?.saveComfyCheckMode,
        loadCatalog: context?.loadCatalog,
        refreshCustomNodesInfoFlow: context?.refreshCustomNodesInfoFlow,
        setExpandedModule: context?.setExpandedModule,
        statusCards: context?.statusCards,
        hasPendingComfyInfoRefresh: context?.hasPendingComfyInfoRefresh,
        loadComfyInfoSnapshot: context?.loadComfyInfoSnapshot,
        renderComfyAlert: context?.renderComfyAlert,
        shouldContinue: context?.isPickerAlive,
        setStartupBusy: context?.setStartupBusy,
        startCatalogStartupLoad: (options = {}) => context?.startCatalogStartupLoad?.(options),
        hasPendingCustomRefresh: context?.hasPendingCustomRefresh,
        hasPendingUpdate: context?.hasPendingUpdate,
        resumePendingCustomRefreshFlow: context?.resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow: context?.resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow: context?.resumePendingComfyInfoRefreshFlow,
    };
}
