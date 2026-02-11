/**
 * Module: web/orchestration/module_node_picker_bindings.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Event bindings and startup-load helpers for Module Node Picker.
 *
 * Purpose:
 *   Keeps UI wiring (onchange/onclick) out of the main picker render module.
 */

/**
 * Bind picker UI events to passed callbacks/state helpers.
 */
export function bindModuleNodePickerEvents(context) {
    const groupSelect = context?.groupSelect;
    const categorySelect = context?.categorySelect;
    const moduleFilter = context?.moduleFilter;
    const nodeSelect = context?.nodeSelect;
    const nodeList = context?.nodeList;
    const updateAllBtn = context?.updateAllBtn;
    const comfyUpdateBtn = context?.comfyUpdateBtn;
    const comfyInstallReqBtn = context?.comfyInstallReqBtn;
    const comfyInfoBtn = context?.comfyInfoBtn;
    const comfyModeSelect = context?.comfyModeSelect;
    const refreshBtn = context?.refreshBtn;
    const isCustomCategory = context?.isCustomCategory;
    const pickerStore = context?.pickerStore;
    const getSelectedGroup = context?.getSelectedGroup;
    const fillModuleSelect = context?.fillModuleSelect;
    const syncUpdateAllButton = context?.syncUpdateAllButton;
    const syncPickerSelectionState = context?.syncPickerSelectionState;
    const loadModuleInfo = context?.loadModuleInfo;
    const isActionBusy = context?.isActionBusy;
    const setCustomStatusChecked = context?.setCustomStatusChecked;
    const setProcessTarget = context?.setProcessTarget;
    const runModuleUpdate = context?.runModuleUpdate;
    const installComfyUIRequirementsFlow = context?.installComfyUIRequirementsFlow;
    const refreshComfyUIInfoFlow = context?.refreshComfyUIInfoFlow;
    const saveComfyCheckMode = context?.saveComfyCheckMode;
    const loadCatalog = context?.loadCatalog;
    const refreshCustomNodesInfoFlow = context?.refreshCustomNodesInfoFlow;
    const setExpandedModule = context?.setExpandedModule;
    let comfyModeReloadTimer = 0;
    let comfyModeReloadToken = 0;

    if (!groupSelect || !categorySelect || !moduleFilter || !nodeSelect) {
        return;
    }

    groupSelect.onchange = () => {
        if (typeof isCustomCategory === "function" && isCustomCategory()) {
            return;
        }
        pickerStore?.set?.({ selectedGroup: getSelectedGroup?.() || "custom" });
        fillModuleSelect?.();
        syncUpdateAllButton?.();
    };

    categorySelect.onchange = () => {
        if (groupSelect) {
            groupSelect.style.display = (typeof isCustomCategory === "function" && isCustomCategory()) ? "none" : "";
        }
        pickerStore?.set?.({ selectedGroup: getSelectedGroup?.() || "custom" });
        fillModuleSelect?.();
        syncUpdateAllButton?.();
    };

    moduleFilter.oninput = () => fillModuleSelect?.();

    nodeSelect.onchange = () => {
        setExpandedModule?.("");
        if (nodeList) {
            nodeList.innerHTML = "";
        }
        syncPickerSelectionState?.();
        loadModuleInfo?.();
    };

    if (updateAllBtn) {
        updateAllBtn.onclick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            setCustomStatusChecked?.(true);
            setProcessTarget?.("custom");
            await runModuleUpdate?.("all", "");
        };
    }

    if (comfyUpdateBtn) {
        comfyUpdateBtn.onclick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            setProcessTarget?.("comfy");
            await runModuleUpdate?.("comfyui", "");
        };
    }

    if (comfyInstallReqBtn) {
        comfyInstallReqBtn.onclick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            await installComfyUIRequirementsFlow?.();
        };
    }

    if (comfyInfoBtn) {
        comfyInfoBtn.onclick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            await refreshComfyUIInfoFlow?.();
        };
    }

    if (comfyModeSelect) {
        comfyModeSelect.onchange = async () => {
            saveComfyCheckMode?.(comfyModeSelect.value);
            const token = ++comfyModeReloadToken;
            if (comfyModeReloadTimer) {
                window.clearTimeout(comfyModeReloadTimer);
            }
            comfyModeReloadTimer = window.setTimeout(async () => {
                if (token !== comfyModeReloadToken) {
                    return;
                }
                await loadCatalog?.();
            }, 120);
        };
    }

    if (refreshBtn) {
        refreshBtn.onclick = async () => {
            setCustomStatusChecked?.(true);
            await refreshCustomNodesInfoFlow?.();
        };
    }
}

/**
 * Perform startup catalog load using persisted picker selection.
 */
export function runModuleNodePickerStartupLoad(context) {
    const pickerStore = context?.pickerStore;
    const defaultModule = String(context?.defaultModule || "");
    const loadCatalog = context?.loadCatalog;
    const startupGroup = String(pickerStore?.get?.("selectedGroup") || "custom").trim();
    const startupModule = String(pickerStore?.get?.("selectedModule") || defaultModule).trim();
    loadCatalog?.({
        preferredGroup: startupGroup || "custom",
        preferredModule: startupModule || defaultModule,
    });
}
