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
    const refreshComfyUIModeInfoFlow = context?.refreshComfyUIModeInfoFlow;
    const saveComfyCheckMode = context?.saveComfyCheckMode;
    const loadCatalog = context?.loadCatalog;
    const refreshCustomNodesInfoFlow = context?.refreshCustomNodesInfoFlow;
    const setExpandedModule = context?.setExpandedModule;
    let comfyModeReloadTimer = 0;
    let comfyModeReloadToken = 0;

    if (!groupSelect || !categorySelect || !moduleFilter || !nodeSelect) {
        return () => {};
    }

    const onGroupChange = () => {
        if (typeof isCustomCategory === "function" && isCustomCategory()) {
            return;
        }
        pickerStore?.set?.({ selectedGroup: getSelectedGroup?.() || "custom" });
        fillModuleSelect?.();
        syncUpdateAllButton?.();
    };
    groupSelect.onchange = onGroupChange;

    const onCategoryChange = () => {
        if (groupSelect) {
            groupSelect.style.display = (typeof isCustomCategory === "function" && isCustomCategory()) ? "none" : "";
        }
        pickerStore?.set?.({ selectedGroup: getSelectedGroup?.() || "custom" });
        fillModuleSelect?.();
        syncUpdateAllButton?.();
    };
    categorySelect.onchange = onCategoryChange;

    const onModuleFilterInput = () => fillModuleSelect?.();
    moduleFilter.oninput = onModuleFilterInput;

    const onNodeSelectChange = () => {
        setExpandedModule?.("");
        if (nodeList) {
            nodeList.innerHTML = "";
        }
        syncPickerSelectionState?.();
        loadModuleInfo?.();
    };
    nodeSelect.onchange = onNodeSelectChange;

    let onUpdateAllClick = null;
    if (updateAllBtn) {
        onUpdateAllClick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            setCustomStatusChecked?.(true);
            setProcessTarget?.("custom");
            await runModuleUpdate?.("all", "");
        };
        updateAllBtn.onclick = onUpdateAllClick;
    }

    let onComfyUpdateClick = null;
    if (comfyUpdateBtn) {
        onComfyUpdateClick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            setProcessTarget?.("comfy");
            await runModuleUpdate?.("comfyui", "");
        };
        comfyUpdateBtn.onclick = onComfyUpdateClick;
    }

    let onComfyInstallReqClick = null;
    if (comfyInstallReqBtn) {
        onComfyInstallReqClick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            await installComfyUIRequirementsFlow?.();
        };
        comfyInstallReqBtn.onclick = onComfyInstallReqClick;
    }

    let onComfyInfoClick = null;
    if (comfyInfoBtn) {
        onComfyInfoClick = async () => {
            if (isActionBusy?.()) {
                return;
            }
            await refreshComfyUIInfoFlow?.();
        };
        comfyInfoBtn.onclick = onComfyInfoClick;
    }

    let onComfyModeChange = null;
    if (comfyModeSelect) {
        onComfyModeChange = async () => {
            saveComfyCheckMode?.(comfyModeSelect.value);
            const token = ++comfyModeReloadToken;
            if (comfyModeReloadTimer) {
                window.clearTimeout(comfyModeReloadTimer);
            }
            comfyModeReloadTimer = window.setTimeout(async () => {
                if (token !== comfyModeReloadToken) {
                    return;
                }
                if (typeof refreshComfyUIModeInfoFlow === "function") {
                    await refreshComfyUIModeInfoFlow();
                    return;
                }
                await loadCatalog?.();
            }, 120);
        };
        comfyModeSelect.onchange = onComfyModeChange;
    }

    let onRefreshClick = null;
    if (refreshBtn) {
        onRefreshClick = async () => {
            setCustomStatusChecked?.(true);
            await refreshCustomNodesInfoFlow?.();
        };
        refreshBtn.onclick = onRefreshClick;
    }

    return () => {
        comfyModeReloadToken += 1;
        if (comfyModeReloadTimer) {
            window.clearTimeout(comfyModeReloadTimer);
            comfyModeReloadTimer = 0;
        }
        if (groupSelect.onchange === onGroupChange) {
            groupSelect.onchange = null;
        }
        if (categorySelect.onchange === onCategoryChange) {
            categorySelect.onchange = null;
        }
        if (moduleFilter.oninput === onModuleFilterInput) {
            moduleFilter.oninput = null;
        }
        if (nodeSelect.onchange === onNodeSelectChange) {
            nodeSelect.onchange = null;
        }
        if (updateAllBtn && updateAllBtn.onclick === onUpdateAllClick) {
            updateAllBtn.onclick = null;
        }
        if (comfyUpdateBtn && comfyUpdateBtn.onclick === onComfyUpdateClick) {
            comfyUpdateBtn.onclick = null;
        }
        if (comfyInstallReqBtn && comfyInstallReqBtn.onclick === onComfyInstallReqClick) {
            comfyInstallReqBtn.onclick = null;
        }
        if (comfyInfoBtn && comfyInfoBtn.onclick === onComfyInfoClick) {
            comfyInfoBtn.onclick = null;
        }
        if (comfyModeSelect && comfyModeSelect.onchange === onComfyModeChange) {
            comfyModeSelect.onchange = null;
        }
        if (refreshBtn && refreshBtn.onclick === onRefreshClick) {
            refreshBtn.onclick = null;
        }
    };
}

/**
 * Perform startup catalog load using persisted picker selection.
 */
export function runModuleNodePickerStartupLoad(context) {
    const pickerStore = context?.pickerStore;
    const defaultModule = String(context?.defaultModule || "");
    const loadCatalog = context?.loadCatalog;
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const maxRetries = Math.max(0, Number(context?.startupRetries ?? 2));
    const retryDelayMs = Math.max(50, Number(context?.startupRetryDelayMs ?? 250));
    const startupGroup = String(pickerStore?.get?.("selectedGroup") || "custom").trim();
    const startupModule = String(pickerStore?.get?.("selectedModule") || defaultModule).trim();
    let cancelled = false;
    let retryTimer = 0;

    const clearRetryTimer = () => {
        if (retryTimer) {
            window.clearTimeout(retryTimer);
            retryTimer = 0;
        }
    };

    const shouldRetryResult = (result) => {
        if (!result || result.ok === false) {
            return true;
        }
        const totalNodes = Number(result.totalNodes || 0);
        const totalModules = Number(result.totalModules || 0);
        return totalNodes <= 0 && totalModules <= 0;
    };

    const runAttempt = async (attempt) => {
        if (cancelled || !shouldContinue()) {
            return;
        }
        const result = await loadCatalog?.({
            preferredGroup: startupGroup || "custom",
            preferredModule: startupModule || defaultModule,
        });
        if (cancelled || !shouldContinue()) {
            return;
        }
        if (attempt >= maxRetries) {
            return;
        }
        if (!shouldRetryResult(result)) {
            return;
        }
        retryTimer = window.setTimeout(() => {
            runAttempt(attempt + 1);
        }, retryDelayMs);
    };

    runAttempt(0);
    return () => {
        cancelled = true;
        clearRetryTimer();
    };
}
