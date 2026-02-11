/**
 * Module: web/orchestration/flow/module_node_picker_catalog_controller.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Catalog/module data controller for Module Node Picker frontend.
 *
 * Purpose:
 *   Owns catalog/module loading tokens, loading guards, and option/diff cache
 *   updates while delegating UI rendering to injected callbacks.
 */

import {
    updateModuleOptionText,
    cacheModuleNodeDiffs,
    loadModuleInfoFlow,
    loadCatalogFlow,
} from "./module_node_picker_data_flow.js";
import { createModuleNodePickerWarmupController } from "../runtime/module_node_picker_warmup_controller.js";

/**
 * Create catalog controller used by picker composition layer.
 */
export function createModuleNodePickerCatalogController(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const getSelectedGroup = typeof context?.getSelectedGroup === "function"
        ? context.getSelectedGroup
        : () => "custom";
    const getSelectedModule = typeof context?.getSelectedModule === "function"
        ? context.getSelectedModule
        : () => "";
    const fetchModuleInfo = context?.fetchModuleInfo;
    const fetchNodeCatalog = context?.fetchNodeCatalog;
    const getComfyMode = typeof context?.getComfyMode === "function"
        ? context.getComfyMode
        : () => "releases";
    const catalogByGroup = context?.catalogByGroup;
    const moduleCounts = context?.moduleCounts;
    const moduleOptions = context?.moduleOptions;
    const moduleBadges = context?.moduleBadges;
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    const formatModuleOption = context?.formatModuleOption;
    const marks = context?.marks || {
        updatedMark: "✅",
        remoteUpdateMark: "🟥",
    };
    const renderNodeList = typeof context?.renderNodeList === "function"
        ? context.renderNodeList
        : () => {};
    const renderModuleInfo = typeof context?.renderModuleInfo === "function"
        ? context.renderModuleInfo
        : () => {};
    const moduleBadgesFromInfo = context?.moduleBadgesFromInfo;
    const setCatalogControlsLoading = typeof context?.setCatalogControlsLoading === "function"
        ? context.setCatalogControlsLoading
        : () => {};
    const setCustomModulesNeedUpdate = typeof context?.setCustomModulesNeedUpdate === "function"
        ? context.setCustomModulesNeedUpdate
        : () => {};
    const setWarmupIndicator = typeof context?.setWarmupIndicator === "function"
        ? context.setWarmupIndicator
        : () => {};
    const renderComfyAlert = typeof context?.renderComfyAlert === "function"
        ? context.renderComfyAlert
        : () => {};
    const fillGroupSelect = typeof context?.fillGroupSelect === "function"
        ? context.fillGroupSelect
        : () => {};
    const groupLabels = context?.groupLabels || {};
    const setHelpText = typeof context?.setHelpText === "function"
        ? context.setHelpText
        : () => {};
    const syncUpdateAllButton = typeof context?.syncUpdateAllButton === "function"
        ? context.syncUpdateAllButton
        : () => {};
    const comfyAlert = context?.comfyAlert || null;
    const comfyAlertText = context?.comfyAlertText || null;
    const comfyUpdateBtn = context?.comfyUpdateBtn || null;
    const comfyInstallReqBtn = context?.comfyInstallReqBtn || null;
    const groupSelect = context?.groupSelect || null;
    const nodeSelect = context?.nodeSelect || null;
    const clearModuleInfo = typeof context?.clearModuleInfo === "function"
        ? context.clearModuleInfo
        : () => {};
    const nodeList = context?.nodeList || null;

    let moduleInfoLoadToken = 0;
    let catalogLoadToken = 0;
    let catalogLoadBusyCount = 0;

    const warmupController = createModuleNodePickerWarmupController({
        shouldContinue,
        setWarmupIndicator,
        maxAttempts: 30,
        delayMs: 1000,
    });

    const getNodesForSelectedGroup = () => {
        const group = getSelectedGroup();
        return catalogByGroup.get(group) || [];
    };

    const setModuleOptionText = (moduleName) => {
        updateModuleOptionText(
            {
                moduleOptions,
                moduleCounts,
                moduleBadges,
                formatModuleOption,
                marks: {
                    updatedMark: marks.updatedMark,
                    remoteUpdateMark: marks.remoteUpdateMark,
                },
            },
            moduleName
        );
    };

    const setModuleNodeDiffs = (moduleName, info) => {
        cacheModuleNodeDiffs(
            {
                moduleNodeDiffs,
            },
            moduleName,
            info
        );
    };

    const loadModuleInfo = async (options = {}) => {
        if (!shouldContinue()) {
            return;
        }
        const token = ++moduleInfoLoadToken;
        return loadModuleInfoFlow(options, {
            isRequestActive: () => token === moduleInfoLoadToken && shouldContinue(),
            getSelectedModule,
            getSelectedGroup,
            fetchModuleInfo,
            clearModuleInfo,
            renderModuleInfo,
            moduleBadgesFromInfo,
            moduleBadges,
            setModuleNodeDiffs,
            setModuleOptionText,
            renderNodeList,
        });
    };

    const loadCatalog = async (options = {}) => {
        if (!shouldContinue()) {
            return;
        }
        const isWarmupPoll = Boolean(options?.warmupPoll);
        const token = ++catalogLoadToken;
        if (!isWarmupPoll) {
            warmupController.onManualLoadStart();
            catalogLoadBusyCount += 1;
            if (catalogLoadBusyCount === 1) {
                setCatalogControlsLoading(true);
            }
        }
        try {
            const result = await loadCatalogFlow(options, {
                isRequestActive: () => token === catalogLoadToken && shouldContinue(),
                fetchNodeCatalog,
                getComfyMode,
                catalogByGroup,
                setCustomModulesNeedUpdate: (value) => {
                    setCustomModulesNeedUpdate(value);
                },
                renderComfyAlert,
                fillGroupSelect,
                groupLabels,
                setHelpText,
                syncUpdateAllButton,
                comfyAlert,
                comfyAlertText,
                comfyUpdateBtn,
                comfyInstallReqBtn,
                groupSelect,
                nodeSelect,
                clearModuleInfo,
                nodeList,
            });
            if (token !== catalogLoadToken || !shouldContinue()) {
                return result;
            }
            warmupController.handleCatalogResult(result, () => ({
                warmupPoll: true,
                preferredGroup: getSelectedGroup(),
                preferredModule: getSelectedModule(),
            }));
            return result;
        } finally {
            if (!isWarmupPoll) {
                catalogLoadBusyCount = Math.max(0, catalogLoadBusyCount - 1);
                if (catalogLoadBusyCount === 0) {
                    setCatalogControlsLoading(false);
                }
            }
        }
    };

    // Wire warmup poller to catalog loader so background retries actually run.
    warmupController.setPoller((nextOptions = {}) => loadCatalog(nextOptions));

    const bumpRequestTokens = () => {
        moduleInfoLoadToken += 1;
        catalogLoadToken += 1;
        warmupController.dispose();
    };

    return {
        getNodesForSelectedGroup,
        setModuleOptionText,
        setModuleNodeDiffs,
        loadModuleInfo,
        loadCatalog,
        bumpRequestTokens,
    };
}
