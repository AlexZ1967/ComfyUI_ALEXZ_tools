/**
 * Module: web/orchestration/module_node_picker_action_flows.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Action-flow orchestration for Module Node Picker.
 *
 * Purpose:
 *   Composes refresh/update/install/resume flows into one factory so the main
 *   picker module only wires dependencies and does not duplicate orchestration code.
 */

import {
    runInstallComfyUIRequirementsFlow,
    maybeInstallChangedRequirementsFlow,
    runModuleUpdateFlow,
} from "./module_node_picker_update_flow.js";
import {
    runRefreshModuleInfoAction,
    runInstallSingleModuleRequirementsAction,
    runRefreshComfyUIInfoAction,
    runRefreshCustomNodesInfoAction,
} from "./module_node_picker_actions.js";
import {
    resumePendingCustomRefreshFlow as resumePendingCustomRefreshFlowImpl,
    resumePendingModuleUpdateFlow as resumePendingModuleUpdateFlowImpl,
    resumePendingComfyInfoRefreshFlow as resumePendingComfyInfoRefreshFlowImpl,
} from "./module_node_picker_resume_flow.js";

/**
 * Create composed long-running action flows used by Module Node Picker.
 */
export function createModuleNodePickerActionFlows(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const setActionBusy = context?.setActionBusy || (() => {});
    const setProcessTarget = context?.setProcessTarget || (() => {});
    const setProcessAction = context?.setProcessAction || (() => {});
    const setRefreshLine = context?.setRefreshLine || (() => {});
    const setCustomRefreshCardLine = context?.setCustomRefreshCardLine || (() => {});
    const pollRefreshProgress = context?.pollRefreshProgress || (async () => null);
    const pollUpdateProgress = context?.pollUpdateProgress || (async () => null);
    const startModuleUpdate = context?.startModuleUpdate || (async () => ({}));
    const installModuleRequirements = context?.installModuleRequirements || (async () => ({}));
    const installComfyUIRequirements = context?.installComfyUIRequirements || (async () => ({}));
    const fetchComfyUIInfo = context?.fetchComfyUIInfo || (async () => ({}));
    const refreshModuleRuntimeState = context?.refreshModuleRuntimeState || (async () => ({}));
    const acknowledgeAllModuleNovelty = context?.acknowledgeAllModuleNovelty || (async () => ({}));
    const fetchModuleRefreshStatus = context?.fetchModuleRefreshStatus || (async () => ({}));
    const fetchModuleUpdateStatus = context?.fetchModuleUpdateStatus || (async () => ({}));
    const getComfyMode = typeof context?.getComfyMode === "function"
        ? context.getComfyMode
        : () => "releases";
    const getLogMode = typeof context?.getLogMode === "function"
        ? context.getLogMode
        : () => "summary";
    const getSelectedGroup = typeof context?.getSelectedGroup === "function"
        ? context.getSelectedGroup
        : () => "";
    const getSelectedModule = typeof context?.getSelectedModule === "function"
        ? context.getSelectedModule
        : () => "";
    const getLoadCatalog = typeof context?.getLoadCatalog === "function"
        ? context.getLoadCatalog
        : () => async () => {};
    const getLoadModuleInfo = typeof context?.getLoadModuleInfo === "function"
        ? context.getLoadModuleInfo
        : () => async () => {};
    const syncUpdateAllButton = context?.syncUpdateAllButton || (() => {});
    const setModuleInlineStatus = context?.setModuleInlineStatus || (() => {});
    const renderComfyAlert = context?.renderComfyAlert || (() => {});
    const setCustomStatusChecked = context?.setCustomStatusChecked || (() => {});
    const setComfyStatusChecked = context?.setComfyStatusChecked || (() => {});
    const setPendingUpdate = context?.setPendingUpdate || (() => {});
    const clearPendingUpdate = context?.clearPendingUpdate || (() => {});
    const setPendingCustomRefresh = context?.setPendingCustomRefresh || (() => {});
    const clearPendingCustomRefresh = context?.clearPendingCustomRefresh || (() => {});
    const setPendingComfyInfoRefresh = context?.setPendingComfyInfoRefresh || (() => {});
    const clearPendingComfyInfoRefresh = context?.clearPendingComfyInfoRefresh || (() => {});
    const hasPendingCustomRefresh = context?.hasPendingCustomRefresh || (() => false);
    const hasPendingUpdate = context?.hasPendingUpdate || (() => false);
    const hasPendingComfyInfoRefresh = context?.hasPendingComfyInfoRefresh || (() => false);
    const onMarkUpdatedModule = context?.onMarkUpdatedModule || (() => {});
    const isModuleMarkedUpdated = context?.isModuleMarkedUpdated || (() => false);
    const comfyAlert = context?.comfyAlert || null;
    const comfyAlertText = context?.comfyAlertText || null;
    const customAlert = context?.customAlert || null;
    const customAlertText = context?.customAlertText || null;
    const formatRefreshLine = context?.formatRefreshLine || (() => "");
    const formatUpdateLine = context?.formatUpdateLine || (() => "");
    const isCanceledRequestError = context?.isCanceledRequestError || (() => false);

    const installComfyUIRequirementsFlow = async () => {
        return runInstallComfyUIRequirementsFlow({
            shouldContinue,
            setActionBusy,
            setProcessTarget,
            setRefreshLine,
            installComfyUIRequirements,
            fetchComfyUIInfo,
            getComfyMode,
            renderComfyAlert,
            setProcessAction,
            syncUpdateAllButton,
        });
    };

    const maybeInstallChangedRequirements = async (update) => {
        return maybeInstallChangedRequirementsFlow(update, {
            shouldContinue,
            setRefreshLine,
            setProcessAction,
            installComfyUIRequirementsFlow,
            installModuleRequirements,
            setActionBusy,
        });
    };

    const runModuleUpdate = async (scope, moduleName) => {
        return runModuleUpdateFlow(scope, moduleName, {
            shouldContinue,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            startModuleUpdate,
            getLogMode,
            pollUpdateProgress,
            getSelectedGroup,
            getSelectedModule,
            onMarkUpdatedModule,
            isModuleMarkedUpdated,
            maybeInstallChangedRequirements,
            loadCatalog: getLoadCatalog(),
            loadModuleInfo: getLoadModuleInfo(),
            syncUpdateAllButton,
            setPendingUpdate,
            clearPendingUpdate,
        });
    };

    const refreshModuleInfoFlow = async (moduleName, syncUpstream) => {
        return runRefreshModuleInfoAction(moduleName, syncUpstream, {
            shouldContinue,
            setProcessTarget,
            setRefreshLine,
            setProcessAction,
            setModuleInlineStatus,
            setActionBusy,
            loadModuleInfo: getLoadModuleInfo(),
            syncUpdateAllButton,
        });
    };

    const installSingleModuleRequirementsFlow = async (moduleName) => {
        return runInstallSingleModuleRequirementsAction(moduleName, {
            shouldContinue,
            setProcessTarget,
            setRefreshLine,
            setProcessAction,
            setModuleInlineStatus,
            setActionBusy,
            installModuleRequirements,
            loadModuleInfo: getLoadModuleInfo(),
            syncUpdateAllButton,
        });
    };

    const refreshComfyUIInfoFlow = async () => {
        return runRefreshComfyUIInfoAction({
            shouldContinue,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            comfyAlert,
            comfyAlertText,
            fetchComfyUIInfo,
            getComfyMode,
            getLogMode,
            renderComfyAlert,
            syncUpdateAllButton,
            setComfyStatusChecked,
            setPendingComfyInfoRefresh,
            clearPendingComfyInfoRefresh,
        });
    };

    const refreshCustomNodesInfoFlow = async () => {
        return runRefreshCustomNodesInfoAction({
            shouldContinue,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            customAlert,
            customAlertText,
            refreshModuleRuntimeState,
            getLogMode,
            pollRefreshProgress,
            acknowledgeAllModuleNovelty,
            loadCatalog: getLoadCatalog(),
            setCustomStatusChecked,
            setPendingCustomRefresh,
            clearPendingCustomRefresh,
        });
    };

    const resumePendingCustomRefreshFlow = async () => {
        return resumePendingCustomRefreshFlowImpl({
            hasPendingCustomRefresh,
            shouldContinue,
            setCustomStatusChecked,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            setCustomRefreshCardLine,
            fetchModuleRefreshStatus,
            pollRefreshProgress,
            acknowledgeAllModuleNovelty,
            loadCatalog: getLoadCatalog(),
            clearPendingCustomRefresh,
            formatRefreshLine,
            isCanceledRequestError,
        });
    };

    const resumePendingModuleUpdateFlow = async () => {
        return resumePendingModuleUpdateFlowImpl({
            hasPendingUpdate,
            shouldContinue,
            setActionBusy,
            setProcessAction,
            setRefreshLine,
            fetchModuleUpdateStatus,
            setProcessTarget,
            formatUpdateLine,
            pollUpdateProgress,
            clearPendingUpdate,
            maybeInstallChangedRequirements,
            loadCatalog: getLoadCatalog(),
            loadModuleInfo: getLoadModuleInfo(),
            isCanceledRequestError,
        });
    };

    const resumePendingComfyInfoRefreshFlow = async () => {
        return resumePendingComfyInfoRefreshFlowImpl({
            hasPendingComfyInfoRefresh,
            shouldContinue,
            setActionBusy,
            setProcessTarget,
            setProcessAction,
            setRefreshLine,
            comfyAlert,
            comfyAlertText,
            fetchComfyUIInfo,
            getComfyMode,
            getLogMode,
            renderComfyAlert,
            clearPendingComfyInfoRefresh,
            syncUpdateAllButton,
            isCanceledRequestError,
        });
    };

    return {
        installComfyUIRequirementsFlow,
        maybeInstallChangedRequirements,
        runModuleUpdate,
        refreshModuleInfoFlow,
        installSingleModuleRequirementsFlow,
        refreshComfyUIInfoFlow,
        refreshCustomNodesInfoFlow,
        resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow,
    };
}
