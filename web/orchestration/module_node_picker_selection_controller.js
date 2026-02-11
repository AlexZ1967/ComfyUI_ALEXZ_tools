/**
 * Module: web/orchestration/module_node_picker_selection_controller.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Selection/dropdown controller for Module Node Picker.
 *
 * Purpose:
 *   Owns category/group/module selection helpers and dropdown population logic
 *   while keeping picker composition module smaller and easier to reason about.
 */

import {
    fillModuleSelectUi,
    fillGroupSelectUi,
} from "../ui/module_node_picker_catalog.js";

/**
 * Create selection controller for picker category/group/module dropdowns.
 */
export function createModuleNodePickerSelectionController(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const categorySelect = context?.categorySelect;
    const groupSelect = context?.groupSelect;
    const nodeSelect = context?.nodeSelect;
    const moduleFilter = context?.moduleFilter;
    const moduleInfo = context?.moduleInfo;
    const nodeList = context?.nodeList;
    const pickerStore = context?.pickerStore;
    const catalogByGroup = context?.catalogByGroup;
    const moduleCatalogByGroup = context?.moduleCatalogByGroup;
    const moduleCounts = context?.moduleCounts;
    const moduleOptions = context?.moduleOptions;
    const moduleBadges = context?.moduleBadges;
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    const moduleBadgesFromModuleEntry = context?.moduleBadgesFromModuleEntry;
    const formatModuleOption = context?.formatModuleOption;
    const marks = context?.marks || {
        updatedMark: "✅",
        remoteUpdateMark: "🟥",
    };
    const defaultModule = context?.defaultModule || "ComfyUI_ALEXZ_tools";
    const comfyGroupOrder = context?.comfyGroupOrder || [];
    const groupLabels = context?.groupLabels || {};
    const setHelpText = typeof context?.setHelpText === "function"
        ? context.setHelpText
        : () => {};
    const syncUpdateAllButton = typeof context?.syncUpdateAllButton === "function"
        ? context.syncUpdateAllButton
        : () => {};
    const setExpandedModule = typeof context?.setExpandedModule === "function"
        ? context.setExpandedModule
        : () => {};
    const getRenderNodeList = typeof context?.getRenderNodeList === "function"
        ? context.getRenderNodeList
        : () => () => {};
    const getLoadModuleInfo = typeof context?.getLoadModuleInfo === "function"
        ? context.getLoadModuleInfo
        : () => async () => {};

    const isCustomCategory = () => String(categorySelect?.value || "") === "custom";

    const getSelectedGroup = () => {
        if (isCustomCategory()) {
            return "custom";
        }
        return String(groupSelect?.value || "").trim();
    };

    const syncPickerSelectionState = () => {
        const partial = {
            selectedGroup: getSelectedGroup() || "custom",
        };
        const selectedModule = String(nodeSelect?.value || "").trim();
        if (selectedModule && selectedModule !== "-1") {
            partial.selectedModule = selectedModule;
        }
        pickerStore?.set?.(partial);
    };

    const getNodesForSelectedGroup = () => {
        const group = getSelectedGroup();
        return catalogByGroup?.get?.(group) || [];
    };

    const fillModuleSelect = (options = {}) => {
        if (!shouldContinue()) {
            return;
        }
        fillModuleSelectUi({
            options,
            nodes: getNodesForSelectedGroup(),
            selectedGroup: getSelectedGroup(),
            moduleEntries: moduleCatalogByGroup?.get?.(getSelectedGroup()) || [],
            moduleFilterValue: moduleFilter?.value || "",
            moduleFilterRaw: moduleFilter?.value || "",
            previousSelectedModule: nodeSelect?.value || "",
            moduleCounts,
            moduleOptions,
            moduleBadges,
            moduleNodeDiffs,
            nodeSelect,
            moduleInfo,
            nodeList,
            pickerStore,
            getSelectedGroup,
            setHelpText,
            syncUpdateAllButton,
            moduleBadgesFromModuleEntry,
            formatModuleOption,
            marks: {
                updatedMark: marks.updatedMark,
                remoteUpdateMark: marks.remoteUpdateMark,
            },
            defaultModule,
            setExpandedModule: (value) => {
                setExpandedModule(value);
            },
            syncPickerSelectionState,
            renderNodeList: () => getRenderNodeList()?.(),
            loadModuleInfo: (...args) => getLoadModuleInfo()?.(...args),
        });
    };

    const fillGroupSelect = (groups, options = {}) => {
        if (!shouldContinue()) {
            return;
        }
        fillGroupSelectUi({
            groups,
            options,
            previousCategory: categorySelect?.value || "",
            previousGroup: groupSelect?.value || "",
            catalogByGroup,
            moduleCatalogByGroup,
            comfyGroupOrder,
            groupLabels,
            groupSelect,
            categorySelect,
            isCustomCategory,
            pickerStore,
            getSelectedGroup,
            fillModuleSelect,
        });
    };

    return {
        isCustomCategory,
        getSelectedGroup,
        syncPickerSelectionState,
        getNodesForSelectedGroup,
        fillModuleSelect,
        fillGroupSelect,
    };
}
