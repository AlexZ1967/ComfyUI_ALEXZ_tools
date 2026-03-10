/**
 * Module: web/orchestration/flow/stage/module_node_picker_flow_stage.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Flow-stage assembly for Module Node Picker composition.
 *
 * Purpose:
 *   Builds polling/catalog/action/module-panel controllers and exposes
 *   stable adapter functions consumed by runtime bootstrap/composer.
 */

import { createModuleNodePickerFlowWiring } from "./module_node_picker_flow_wiring.js";

/**
 * Create flow-stage controllers and normalized adapter surface for composer.
 */
export function createModuleNodePickerFlowStage(context = {}) {
    const flowWiring = createModuleNodePickerFlowWiring({
        shouldContinue: context?.shouldContinue,
        fetchModuleRefreshStatus: context?.fetchModuleRefreshStatus,
        fetchModuleUpdateStatus: context?.fetchModuleUpdateStatus,
        formatRefreshLine: context?.formatRefreshLine,
        formatUpdateLine: context?.formatUpdateLine,
        setRefreshLine: context?.setRefreshLine,
        getProcessTarget: context?.getProcessTarget,
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        getSelectedGroup: context?.getSelectedGroup,
        getSelectedModule: context?.getSelectedModule,
        getSelectedModuleTrimmed: context?.getSelectedModuleTrimmed,
        fetchModuleInfo: context?.fetchModuleInfo,
        fetchNodeCatalog: context?.fetchNodeCatalog,
        getComfyMode: context?.getComfyMode,
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
        setCustomModulesUnknownUpdateModules: context?.setCustomModulesUnknownUpdateModules,
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
        clearModuleInfo: context?.clearModuleInfo,
        nodeList: context?.nodeList,
        setActionBusy: context?.setActionBusy,
        syncBusyUiState: context?.syncBusyUiState,
        resetBusyState: context?.resetBusyState,
        setProcessTarget: context?.setProcessTarget,
        setProcessAction: context?.setProcessAction,
        setCustomRefreshCardLine: context?.setCustomRefreshCardLine,
        setCatalogControlsLoading: context?.setCatalogControlsLoading,
        startModuleUpdate: context?.startModuleUpdate,
        installModuleRequirements: context?.installModuleRequirements,
        installComfyUIRequirements: context?.installComfyUIRequirements,
        fetchComfyUIInfo: context?.fetchComfyUIInfo,
        getLogMode: context?.getLogMode,
        refreshModuleRuntimeState: context?.refreshModuleRuntimeState,
        acknowledgeAllModuleNovelty: context?.acknowledgeAllModuleNovelty,
        clearUpdatedModulesSession: context?.clearUpdatedModulesSession,
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
        createNodeByInfo: context?.createNodeByInfo,
        app: context?.app,
        centerNode: context?.centerNode,
        fmtDate: context?.fmtDate,
        getActionBusy: context?.getActionBusy,
        getNodesForSelectedGroup: context?.getNodesForSelectedGroup,
        getInlineStatus: context?.getInlineStatus,
    });

    const actionFlows = flowWiring.actionFlows;

    return {
        flowWiring,
        pollingController: flowWiring.pollingController,
        catalogController: flowWiring.catalogController,
        modulePanelController: flowWiring.modulePanelController,
        loadModuleInfo: (options = {}) => flowWiring.loadModuleInfo(options),
        loadCatalog: (options = {}) => flowWiring.loadCatalog(options),
        renderNodeList: () => flowWiring.renderNodeList(),
        renderModuleInfo: (info) => flowWiring.renderModuleInfo(info),
        setExpandedModule: (value) => flowWiring.setExpandedModule(value),
        installComfyUIRequirementsFlow: (...args) => actionFlows.installComfyUIRequirementsFlow(...args),
        maybeInstallChangedRequirements: (...args) => actionFlows.maybeInstallChangedRequirements(...args),
        runModuleUpdate: (...args) => actionFlows.runModuleUpdate(...args),
        refreshComfyUIInfoFlow: (...args) => actionFlows.refreshComfyUIInfoFlow(...args),
        refreshCustomNodesInfoFlow: (...args) => actionFlows.refreshCustomNodesInfoFlow(...args),
        refreshModuleInfoFlow: (...args) => actionFlows.refreshModuleInfoFlow(...args),
        installSingleModuleRequirementsFlow: (...args) => actionFlows.installSingleModuleRequirementsFlow(...args),
        resumePendingCustomRefreshFlow: (...args) => actionFlows.resumePendingCustomRefreshFlow(...args),
        resumePendingModuleUpdateFlow: (...args) => actionFlows.resumePendingModuleUpdateFlow(...args),
        resumePendingComfyInfoRefreshFlow: (...args) => actionFlows.resumePendingComfyInfoRefreshFlow(...args),
    };
}
