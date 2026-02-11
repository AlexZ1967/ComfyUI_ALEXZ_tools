/**
 * Module: web/orchestration/ui/module_node_picker_ui_controllers.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   UI/controller composition for Module Node Picker selectors, busy state,
 *   help/status cards, and top-level card adapters.
 *
 * Purpose:
 *   Keep the main picker composer focused on async/update orchestration by
 *   centralizing UI-controller wiring in a dedicated module.
 */

import { createModuleNodePickerSelectionController } from "./module_node_picker_selection_controller.js";
import { createBusyUiController } from "./module_node_picker_busy_ui.js";
import { createModuleNodePickerViewHelpers } from "./module_node_picker_view_helpers.js";
import { createModuleNodePickerStatusCards } from "./module_node_picker_status_cards.js";

/**
 * Create and wire core UI controllers used by the picker composer.
 */
export function createModuleNodePickerUiControllers(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const marks = context?.marks || {
        updatedMark: "✅",
        remoteUpdateMark: "🟥",
    };
    const getCustomModulesNeedUpdate = typeof context?.getCustomModulesNeedUpdate === "function"
        ? context.getCustomModulesNeedUpdate
        : () => 0;
    const getCustomModulesUnknownUpdate = typeof context?.getCustomModulesUnknownUpdate === "function"
        ? context.getCustomModulesUnknownUpdate
        : () => 0;

    let syncUpdateAllButtonImpl = () => {};
    let setHelpTextImpl = () => {};

    const selectionController = createModuleNodePickerSelectionController({
        shouldContinue,
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
        marks,
        defaultModule: context?.defaultModule,
        comfyGroupOrder: context?.comfyGroupOrder,
        groupLabels: context?.groupLabels,
        setHelpText: (...args) => setHelpTextImpl(...args),
        syncUpdateAllButton: () => syncUpdateAllButtonImpl(),
        setExpandedModule: context?.setExpandedModule,
        getRenderNodeList: context?.getRenderNodeList,
        getLoadModuleInfo: context?.getLoadModuleInfo,
    });

    const busyUi = createBusyUiController({
        shouldContinue,
        controls: context?.controls || {},
        getProcessUi: context?.getProcessUi,
    });

    const viewHelpers = createModuleNodePickerViewHelpers({
        shouldContinue,
        processUi: context?.processUi,
        getActionBusy: () => busyUi.getActionBusy(),
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        help: context?.help,
        marks,
    });
    setHelpTextImpl = viewHelpers.setHelpText;

    const statusCards = createModuleNodePickerStatusCards({
        shouldContinue,
        getComfyMode: context?.getComfyMode,
        getActionBusy: () => busyUi.getActionBusy(),
        fmtDate: context?.fmtDate,
        comfyAlert: context?.comfyAlert,
        comfyAlertText: context?.comfyAlertText,
        comfyUpdateBtn: context?.comfyUpdateBtn,
        comfyInstallReqBtn: context?.comfyInstallReqBtn,
        customAlert: context?.customAlert,
        customAlertText: context?.customAlertText,
        updateAllBtn: context?.updateAllBtn,
        getCustomModulesNeedUpdate,
        getCustomModulesUnknownUpdate,
        saveCustomStatusChecked: context?.saveCustomStatusChecked,
        saveComfyStatusChecked: context?.saveComfyStatusChecked,
        saveComfyInfoSnapshot: context?.saveComfyInfoSnapshot,
        initialCustomStatusChecked: context?.initialCustomStatusChecked,
        initialComfyStatusChecked: context?.initialComfyStatusChecked,
    });
    syncUpdateAllButtonImpl = () => statusCards.syncUpdateAllButton();

    return {
        selectionController,
        busyUi,
        viewHelpers,
        statusCards,
        isCustomCategory: () => selectionController.isCustomCategory(),
        getSelectedGroup: () => selectionController.getSelectedGroup(),
        syncPickerSelectionState: () => selectionController.syncPickerSelectionState(),
        getNodesForSelectedGroup: () => selectionController.getNodesForSelectedGroup(),
        fillModuleSelect: (options = {}) => selectionController.fillModuleSelect(options),
        fillGroupSelect: (groups, options = {}) => selectionController.fillGroupSelect(groups, options),
        syncBusyUiState: () => busyUi.syncBusyUiState(),
        setCatalogControlsLoading: (loading) => busyUi.setCatalogControlsLoading(loading),
        setActionBusy: (busy) => busyUi.setActionBusy(busy),
        setStartupBusy: (busy) => busyUi.setStartupBusy(busy),
        renderComfyAlert: (info) => statusCards.renderComfyAlert(info),
        renderCustomAlert: () => statusCards.renderCustomAlert(),
        syncUpdateAllButton: () => statusCards.syncUpdateAllButton(),
        setCustomStatusChecked: (checked) => statusCards.setCustomStatusChecked(checked),
        setComfyStatusChecked: (checked) => statusCards.setComfyStatusChecked(checked),
    };
}
