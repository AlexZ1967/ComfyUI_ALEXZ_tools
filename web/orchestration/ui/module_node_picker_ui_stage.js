/**
 * Module: web/orchestration/ui/module_node_picker_ui_stage.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   UI-stage assembly for Module Node Picker composition.
 *
 * Purpose:
 *   Builds selector/busy/view/status controllers and exposes normalized
 *   adapters so composer code can stay focused on high-level lifecycle flow.
 */

import { createModuleNodePickerUiControllers } from "./module_node_picker_ui_controllers.js";

/**
 * Create UI-stage controllers and normalized adapter surface for composer.
 */
export function createModuleNodePickerUiStage(context = {}) {
    const uiControllers = createModuleNodePickerUiControllers({
        shouldContinue: context?.shouldContinue,
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
    });

    return {
        uiControllers,
        selectionController: uiControllers.selectionController,
        busyUi: uiControllers.busyUi,
        viewHelpers: uiControllers.viewHelpers,
        statusCards: uiControllers.statusCards,
        showHelpStatus: uiControllers.viewHelpers.setHelpText,
        isCustomCategory: uiControllers.isCustomCategory,
        getSelectedGroup: uiControllers.getSelectedGroup,
        syncPickerSelectionState: uiControllers.syncPickerSelectionState,
        getNodesForSelectedGroup: uiControllers.getNodesForSelectedGroup,
        fillModuleSelect: uiControllers.fillModuleSelect,
        fillGroupSelect: uiControllers.fillGroupSelect,
        setCatalogControlsLoading: uiControllers.setCatalogControlsLoading,
        setActionBusy: uiControllers.setActionBusy,
        setStartupBusy: uiControllers.setStartupBusy,
        renderComfyAlert: uiControllers.renderComfyAlert,
        renderCustomAlert: uiControllers.renderCustomAlert,
        syncUpdateAllButton: uiControllers.syncUpdateAllButton,
        setCustomStatusChecked: uiControllers.setCustomStatusChecked,
        setComfyStatusChecked: uiControllers.setComfyStatusChecked,
    };
}
