/**
 * Module: web/orchestration/flow/module_node_picker_module_panel_controller.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Module-card and node-list rendering controller for Module Node Picker.
 *
 * Purpose:
 *   Encapsulates expanded-module UI state and render callbacks so the main
 *   picker composition file does not own panel rendering internals.
 */

import {
    renderNodeListPanel,
    renderModuleInfoCard,
} from "../../ui/module_node_picker_renderers.js";

/**
 * Create controller for module card and node list panel rendering.
 */
export function createModuleNodePickerModulePanelController(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const nodeList = context?.nodeList;
    const nodeSelect = context?.nodeSelect;
    const moduleInfo = context?.moduleInfo;
    const moduleCounts = context?.moduleCounts;
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    const updatedModulesSession = context?.updatedModulesSession;
    const marks = context?.marks || {
        updatedMark: "✅",
        remoteUpdateMark: "🟥",
    };
    const setHelpText = context?.setHelpText || (() => {});
    const setHelpHintText = context?.setHelpHintText || (() => {});
    const setHelpModuleCardHint = context?.setHelpModuleCardHint || (() => {});
    const setHelpModuleSummary = context?.setHelpModuleSummary || (() => {});
    const createNodeByInfo = context?.createNodeByInfo;
    const app = context?.app;
    const centerNode = context?.centerNode;
    const fmtDate = context?.fmtDate;
    const getActionBusy = typeof context?.getActionBusy === "function"
        ? context.getActionBusy
        : () => false;
    const getNodesForSelectedGroup = typeof context?.getNodesForSelectedGroup === "function"
        ? context.getNodesForSelectedGroup
        : () => [];
    const getInlineStatus = typeof context?.getInlineStatus === "function"
        ? context.getInlineStatus
        : () => null;
    const onRefreshModuleInfo = context?.onRefreshModuleInfo || (async () => {});
    const onUpdateModule = context?.onUpdateModule || (async () => {});
    const onInstallModuleRequirements = context?.onInstallModuleRequirements || (async () => {});

    let expandedModule = "";

    const setExpandedModule = (value) => {
        expandedModule = String(value || "").trim();
    };

    const toggleExpandedModule = (value) => {
        const normalized = String(value || "").trim();
        expandedModule = expandedModule === normalized ? "" : normalized;
    };

    const clearExpandedModule = () => {
        expandedModule = "";
    };

    const renderNodeList = () => {
        if (!shouldContinue()) {
            return;
        }
        renderNodeListPanel({
            nodeListEl: nodeList,
            selectedModule: nodeSelect?.value || "",
            getNodesForSelectedGroup,
            expandedModule,
            setHelpText,
            setHelpHintText,
            setHelpModuleCardHint,
            setHelpModuleSummary,
            moduleNodeDiffs,
            marks: {
                updatedMark: marks.updatedMark,
                remoteUpdateMark: marks.remoteUpdateMark,
            },
            createNodeByInfo,
            app,
            centerNode,
        });
    };

    const renderModuleInfo = (info) => {
        if (!shouldContinue()) {
            return;
        }
        const selectedModule = String(nodeSelect?.value || "").trim();
        const nodeCount = moduleCounts?.get(selectedModule) || 0;
        renderModuleInfoCard({
            moduleInfoEl: moduleInfo,
            info,
            selectedModule,
            nodeCount,
            isModuleUpdated:
                Boolean(updatedModulesSession?.has?.(selectedModule))
                || Boolean(info?.updated_between_runs)
                || Boolean(info?.new_module_between_runs),
            actionBusy: getActionBusy(),
            inlineStatus: getInlineStatus(selectedModule),
            fmtDate,
            onExpandModule: (moduleName) => {
                toggleExpandedModule(moduleName);
                renderNodeList();
            },
            onRefreshModuleInfo,
            onUpdateModule,
            onInstallModuleRequirements,
        });
    };

    return {
        setExpandedModule,
        clearExpandedModule,
        renderNodeList,
        renderModuleInfo,
    };
}
