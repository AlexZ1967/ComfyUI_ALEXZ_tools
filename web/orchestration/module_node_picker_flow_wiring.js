/**
 * Module: web/orchestration/module_node_picker_flow_wiring.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Async flow/controller wiring for Module Node Picker.
 *
 * Purpose:
 *   Composes polling, catalog loading, update/refresh action flows, and module
 *   panel rendering into a reusable runtime bundle used by the picker composer.
 */

import { createModuleNodePickerPollingController } from "./module_node_picker_polling_controller.js";
import { createModuleNodePickerCatalogController } from "./module_node_picker_catalog_controller.js";
import { createModuleNodePickerActionFlows } from "./module_node_picker_action_flows.js";
import { createModuleNodePickerModulePanelController } from "./module_node_picker_module_panel_controller.js";

/**
 * Wire polling + catalog + update flows and expose runtime adapters.
 */
export function createModuleNodePickerFlowWiring(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;

    let renderNodeListImpl = () => {};
    let renderModuleInfoImpl = () => {};
    let loadModuleInfoImpl = async () => {};
    let loadCatalogImpl = async () => {};

    const pollingController = createModuleNodePickerPollingController({
        shouldContinue,
        fetchModuleRefreshStatus: context?.fetchModuleRefreshStatus,
        fetchModuleUpdateStatus: context?.fetchModuleUpdateStatus,
        formatRefreshLine: context?.formatRefreshLine,
        formatUpdateLine: context?.formatUpdateLine,
        setRefreshLine: context?.setRefreshLine,
        getProcessTarget: context?.getProcessTarget,
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        refreshSleepMs: 400,
        updateSleepMs: 450,
    });
    const pollRefreshProgress = () => pollingController.pollRefreshProgress();
    const pollUpdateProgress = () => pollingController.pollUpdateProgress();

    const catalogController = createModuleNodePickerCatalogController({
        shouldContinue,
        getSelectedGroup: context?.getSelectedGroup,
        getSelectedModule: context?.getSelectedModule,
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
        renderNodeList: () => renderNodeListImpl(),
        renderModuleInfo: (...args) => renderModuleInfoImpl(...args),
        moduleBadgesFromInfo: context?.moduleBadgesFromInfo,
        setCatalogControlsLoading: context?.setCatalogControlsLoading,
        setCustomModulesNeedUpdate: context?.setCustomModulesNeedUpdate,
        renderComfyAlert: context?.renderComfyAlert,
        fillGroupSelect: (...args) => context?.selectionController?.fillGroupSelect?.(...args),
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
    });

    loadModuleInfoImpl = async (options = {}) => catalogController.loadModuleInfo(options);
    loadCatalogImpl = async (options = {}) => catalogController.loadCatalog(options);

    const actionFlows = createModuleNodePickerActionFlows({
        shouldContinue,
        setActionBusy: context?.setActionBusy,
        setProcessTarget: context?.setProcessTarget,
        setProcessAction: context?.setProcessAction,
        setRefreshLine: context?.setRefreshLine,
        setCustomRefreshCardLine: context?.setCustomRefreshCardLine,
        pollRefreshProgress,
        pollUpdateProgress,
        startModuleUpdate: context?.startModuleUpdate,
        installModuleRequirements: context?.installModuleRequirements,
        installComfyUIRequirements: context?.installComfyUIRequirements,
        fetchComfyUIInfo: context?.fetchComfyUIInfo,
        refreshModuleRuntimeState: context?.refreshModuleRuntimeState,
        acknowledgeAllModuleNovelty: context?.acknowledgeAllModuleNovelty,
        fetchModuleRefreshStatus: context?.fetchModuleRefreshStatus,
        fetchModuleUpdateStatus: context?.fetchModuleUpdateStatus,
        getComfyMode: context?.getComfyMode,
        getSelectedGroup: context?.getSelectedGroup,
        getSelectedModule: context?.getSelectedModuleTrimmed,
        getLoadCatalog: () => loadCatalogImpl,
        getLoadModuleInfo: () => loadModuleInfoImpl,
        syncUpdateAllButton: context?.syncUpdateAllButton,
        setModuleInlineStatus: context?.setModuleInlineStatus,
        renderComfyAlert: context?.renderComfyAlert,
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
        comfyAlert: context?.comfyAlert,
        comfyAlertText: context?.comfyAlertText,
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        formatRefreshLine: context?.formatRefreshLine,
        formatUpdateLine: context?.formatUpdateLine,
        isCanceledRequestError: context?.isCanceledRequestError,
    });

    const modulePanelController = createModuleNodePickerModulePanelController({
        shouldContinue,
        nodeList: context?.nodeList,
        nodeSelect: context?.nodeSelect,
        moduleInfo: context?.moduleInfo,
        moduleCounts: context?.moduleCounts,
        moduleNodeDiffs: context?.moduleNodeDiffs,
        updatedModulesSession: context?.updatedModulesSession,
        marks: context?.marks,
        setHelpText: context?.setHelpText,
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
        onRefreshModuleInfo: (...args) => actionFlows.refreshModuleInfoFlow(...args),
        onUpdateModule: async (moduleName) => actionFlows.runModuleUpdate("single", moduleName),
        onInstallModuleRequirements: (...args) => actionFlows.installSingleModuleRequirementsFlow(...args),
    });

    renderNodeListImpl = () => modulePanelController.renderNodeList();
    renderModuleInfoImpl = (info) => modulePanelController.renderModuleInfo(info);

    return {
        pollingController,
        catalogController,
        actionFlows,
        modulePanelController,
        loadModuleInfo: (options = {}) => loadModuleInfoImpl(options),
        loadCatalog: (options = {}) => loadCatalogImpl(options),
        renderNodeList: () => renderNodeListImpl(),
        renderModuleInfo: (info) => renderModuleInfoImpl(info),
        setExpandedModule: (value) => modulePanelController.setExpandedModule(value),
    };
}
