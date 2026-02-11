/**
 * Module: web/orchestration/runtime/bootstrap/module_node_picker_runtime_bootstrap.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime bootstrap wiring for Module Node Picker.
 *
 * Purpose:
 *   Centralizes event binding, status-card restore on reopen, and startup
 *   coordinator wiring so the picker composer remains focused on composition.
 */

import { bindModuleNodePickerEvents } from "../../core/infra/module_node_picker_bindings.js";
import { runStartupCoordinator } from "./module_node_picker_startup_flow.js";

/**
 * Bind runtime events and startup flows for one picker render instance.
 */
export function initializeModuleNodePickerRuntime(context = {}) {
    const unbindPickerEvents = bindModuleNodePickerEvents({
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
        isActionBusy: context?.isActionBusy,
        setCustomStatusChecked: context?.setCustomStatusChecked,
        setProcessTarget: context?.setProcessTarget,
        runModuleUpdate: context?.runModuleUpdate,
        installComfyUIRequirementsFlow: context?.installComfyUIRequirementsFlow,
        refreshComfyUIInfoFlow: context?.refreshComfyUIInfoFlow,
        saveComfyCheckMode: context?.saveComfyCheckMode,
        loadCatalog: context?.loadCatalog,
        refreshCustomNodesInfoFlow: context?.refreshCustomNodesInfoFlow,
        setExpandedModule: context?.setExpandedModule,
    }) || (() => {});

    // Restore last ComfyUI status card across widget switches in current session.
    if (context?.statusCards?.getComfyStatusChecked?.() && !context?.hasPendingComfyInfoRefresh?.()) {
        const lastComfyInfo = context?.loadComfyInfoSnapshot?.();
        if (lastComfyInfo) {
            context?.renderComfyAlert?.(lastComfyInfo);
        }
    }

    const cancelStartupLoad = runStartupCoordinator({
        shouldContinue: context?.shouldContinue,
        setStartupBusy: context?.setStartupBusy,
        startCatalogStartupLoad: context?.startCatalogStartupLoad,
        hasPendingCustomRefresh: context?.hasPendingCustomRefresh,
        hasPendingUpdate: context?.hasPendingUpdate,
        hasPendingComfyInfoRefresh: context?.hasPendingComfyInfoRefresh,
        resumePendingCustomRefreshFlow: context?.resumePendingCustomRefreshFlow,
        resumePendingModuleUpdateFlow: context?.resumePendingModuleUpdateFlow,
        resumePendingComfyInfoRefreshFlow: context?.resumePendingComfyInfoRefreshFlow,
    });

    return {
        unbindPickerEvents,
        cancelStartupLoad,
    };
}
