/**
 * Module: web/ui/module_node_picker_catalog.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Catalog/group/module selector rendering helpers for Module Node Picker.
 *
 * Purpose:
 *   Isolates selector-population logic from picker orchestration and data loading.
 */

/**
 * Populate module selector for current group with filtering and badge placeholders.
 */
export function fillModuleSelectUi(context) {
    const options = context?.options || {};
    const preferredModule = String(options?.preferredModule || "").trim();
    const autoExpandModule = String(options?.autoExpandModule || "").trim();
    const nodes = Array.isArray(context?.nodes) ? context.nodes : [];
    const selectedGroup = String(context?.selectedGroup || "").trim();
    const moduleEntries = Array.isArray(context?.moduleEntries) ? context.moduleEntries : [];
    const filterValue = String(context?.moduleFilterValue || "").trim().toLowerCase();
    const previousSelectedModule = String(context?.previousSelectedModule || "").trim();
    const moduleCounts = context?.moduleCounts;
    const moduleOptions = context?.moduleOptions;
    const moduleBadges = context?.moduleBadges;
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    const nodeSelect = context?.nodeSelect;
    const moduleInfo = context?.moduleInfo;
    const nodeList = context?.nodeList;
    const pickerStore = context?.pickerStore;
    const getSelectedGroup = context?.getSelectedGroup;
    const setHelpText = context?.setHelpText;
    const syncUpdateAllButton = context?.syncUpdateAllButton;
    const moduleFilterRaw = String(context?.moduleFilterRaw || "");
    const moduleBadgesFromModuleEntry = context?.moduleBadgesFromModuleEntry;
    const formatModuleOption = context?.formatModuleOption;
    const marks = context?.marks || {};
    const defaultModule = String(context?.defaultModule || "");
    const setExpandedModule = context?.setExpandedModule;
    const syncPickerSelectionState = context?.syncPickerSelectionState;
    const renderNodeList = context?.renderNodeList;
    const loadModuleInfo = context?.loadModuleInfo;

    if (!moduleCounts || !moduleOptions || !moduleBadges || !moduleNodeDiffs || !nodeSelect) {
        return;
    }

    moduleCounts.clear();
    moduleOptions.clear();
    moduleBadges.clear();
    moduleNodeDiffs.clear();
    nodeSelect.innerHTML = "";

    const grouped = new Map();
    for (const node of nodes) {
        const moduleName = node.module || "unknown";
        if (!grouped.has(moduleName)) {
            grouped.set(moduleName, []);
        }
        grouped.get(moduleName).push(node);
    }

    let modules = [];
    if (moduleEntries.length) {
        modules = moduleEntries
            .map((entry) => String(entry?.module || "unknown"))
            .sort((a, b) => a.localeCompare(b));
    } else {
        modules = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
    }
    if (filterValue) {
        modules = modules.filter((name) => name.toLowerCase().includes(filterValue));
    }

    if (modules.length === 0) {
        const empty = document.createElement("option");
        empty.value = "-1";
        empty.textContent = filterValue ? "Нет модулей по фильтру" : "В этой группе нет модулей";
        nodeSelect.appendChild(empty);
        nodeSelect.value = "-1";
        if (pickerStore && typeof pickerStore.set === "function") {
            const groupForStore = typeof getSelectedGroup === "function" ? getSelectedGroup() : selectedGroup || "custom";
            pickerStore.set({ selectedGroup: groupForStore || "custom" });
        }
        if (moduleInfo) {
            moduleInfo.innerHTML = "";
        }
        if (nodeList) {
            nodeList.innerHTML = "";
        }
        if (typeof setHelpText === "function") {
            setHelpText(filterValue
                ? `Нет модулей по фильтру: "${moduleFilterRaw}".`
                : "Модули не найдены для выбранной группы.");
        }
        syncUpdateAllButton?.();
        return;
    }

    const countMap = new Map();
    const entryMap = new Map();
    for (const entry of moduleEntries) {
        const moduleName = String(entry?.module || "unknown");
        countMap.set(moduleName, Number(entry?.count) || 0);
        entryMap.set(moduleName, entry || {});
    }

    for (const moduleName of modules) {
        const opt = document.createElement("option");
        opt.value = moduleName;
        const count = countMap.has(moduleName)
            ? (countMap.get(moduleName) || 0)
            : (grouped.get(moduleName) || []).length;
        moduleCounts.set(moduleName, count);
        moduleOptions.set(moduleName, opt);
        const entry = entryMap.get(moduleName) || null;
        const badges = typeof moduleBadgesFromModuleEntry === "function"
            ? moduleBadgesFromModuleEntry(entry)
            : null;
        if (badges?.updatedBetweenRuns || badges?.hasRemoteUpdate || badges?.hasUnknownUpdate) {
            moduleBadges.set(moduleName, badges);
        }
        if (typeof formatModuleOption === "function") {
            opt.textContent = formatModuleOption(moduleName, count, badges, marks);
        } else {
            opt.textContent = `${moduleName} (${count})`;
        }
        nodeSelect.appendChild(opt);
    }

    if (preferredModule && modules.includes(preferredModule)) {
        nodeSelect.value = preferredModule;
    } else if (previousSelectedModule && modules.includes(previousSelectedModule)) {
        nodeSelect.value = previousSelectedModule;
    } else if (defaultModule && modules.includes(defaultModule)) {
        nodeSelect.value = defaultModule;
    } else {
        nodeSelect.value = modules[0];
    }

    if (autoExpandModule && nodeSelect.value === autoExpandModule) {
        setExpandedModule?.(autoExpandModule);
    } else {
        setExpandedModule?.("");
        if (nodeList) {
            nodeList.innerHTML = "";
        }
    }

    syncPickerSelectionState?.();
    renderNodeList?.();
    loadModuleInfo?.();
    syncUpdateAllButton?.();
}

/**
 * Populate top-level group selector and propagate selection to module list.
 */
export function fillGroupSelectUi(context) {
    const groups = Array.isArray(context?.groups) ? context.groups : [];
    const options = context?.options || {};
    const preferredGroup = String(options?.preferredGroup || "").trim();
    const preferredModule = String(options?.preferredModule || "").trim();
    const autoExpandModule = String(options?.autoExpandModule || "").trim();
    const previousCategory = String(context?.previousCategory || "").trim();
    const previousGroup = String(context?.previousGroup || "").trim();
    const catalogByGroup = context?.catalogByGroup;
    const moduleCatalogByGroup = context?.moduleCatalogByGroup;
    const comfyGroupOrder = Array.isArray(context?.comfyGroupOrder) ? context.comfyGroupOrder : [];
    const groupLabels = context?.groupLabels || {};
    const groupSelect = context?.groupSelect;
    const categorySelect = context?.categorySelect;
    const isCustomCategory = context?.isCustomCategory;
    const pickerStore = context?.pickerStore;
    const getSelectedGroup = context?.getSelectedGroup;
    const fillModuleSelect = context?.fillModuleSelect;

    if (!catalogByGroup || !moduleCatalogByGroup || !groupSelect || !categorySelect) {
        return;
    }

    moduleCatalogByGroup.clear();
    groups.forEach((group) => {
        catalogByGroup.set(group.id, group.nodes || []);
        moduleCatalogByGroup.set(group.id, group.modules || []);
    });

    const comfyGroups = comfyGroupOrder.filter((groupId) => catalogByGroup.has(groupId));
    groupSelect.innerHTML = "";
    for (const groupId of comfyGroups) {
        const opt = document.createElement("option");
        const nodes = catalogByGroup.get(groupId) || [];
        opt.value = groupId;
        opt.textContent = `${groupLabels[groupId] || groupId} (${nodes.length})`;
        groupSelect.appendChild(opt);
    }

    if (preferredGroup === "custom") {
        categorySelect.value = "custom";
    } else if (preferredGroup && comfyGroupOrder.includes(preferredGroup) && catalogByGroup.has(preferredGroup)) {
        categorySelect.value = "comfy";
    } else if (previousCategory === "comfy" || previousCategory === "custom") {
        categorySelect.value = previousCategory;
    } else if (catalogByGroup.has("custom")) {
        categorySelect.value = "custom";
    } else {
        categorySelect.value = "comfy";
    }

    if (typeof isCustomCategory === "function" && !isCustomCategory()) {
        if (preferredGroup && comfyGroups.includes(preferredGroup)) {
            groupSelect.value = preferredGroup;
        } else if (previousGroup && comfyGroups.includes(previousGroup)) {
            groupSelect.value = previousGroup;
        } else if (comfyGroups.length > 0) {
            groupSelect.value = comfyGroups[0];
        }
    }

    if (typeof isCustomCategory === "function") {
        groupSelect.style.display = isCustomCategory() ? "none" : "";
    }
    if (pickerStore && typeof pickerStore.set === "function") {
        const selected = typeof getSelectedGroup === "function" ? getSelectedGroup() : "custom";
        pickerStore.set({ selectedGroup: selected || "custom" });
    }

    fillModuleSelect?.({ preferredModule, autoExpandModule });
}
