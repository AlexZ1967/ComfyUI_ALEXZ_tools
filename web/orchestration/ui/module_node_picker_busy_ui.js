/**
 * Module: web/orchestration/ui/module_node_picker_busy_ui.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Busy/loading UI state controller for Module Node Picker.
 *
 * Purpose:
 *   Provides one place to manage disable/enable semantics for picker controls
 *   across startup, long-running actions, and catalog loading states.
 */

/**
 * Create busy/loading UI controller for picker controls.
 */
export function createBusyUiController(context) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const controls = context?.controls || {};
    const getProcessUi = typeof context?.getProcessUi === "function"
        ? context.getProcessUi
        : () => null;

    let actionBusy = false;
    let startupBusy = false;
    let catalogControlsLoading = false;

    const syncBusyUiState = () => {
        if (!shouldContinue()) {
            return;
        }
        const busy = Boolean(actionBusy || startupBusy);
        const controlsBusy = Boolean(busy || catalogControlsLoading);

        const refreshBtn = controls.refreshBtn;
        const comfyInfoBtn = controls.comfyInfoBtn;
        const comfyModeSelect = controls.comfyModeSelect;
        const categorySelect = controls.categorySelect;
        const groupSelect = controls.groupSelect;
        const nodeSelect = controls.nodeSelect;
        const moduleFilter = controls.moduleFilter;
        const moduleInfo = controls.moduleInfo;
        const nodeList = controls.nodeList;

        if (refreshBtn) {
            refreshBtn.disabled = busy;
        }
        if (comfyInfoBtn) {
            comfyInfoBtn.disabled = busy;
        }
        if (comfyModeSelect) {
            comfyModeSelect.disabled = busy;
        }
        if (categorySelect) {
            categorySelect.disabled = controlsBusy;
        }
        if (groupSelect) {
            groupSelect.disabled = controlsBusy;
        }
        if (nodeSelect) {
            nodeSelect.disabled = controlsBusy;
        }
        if (moduleFilter) {
            moduleFilter.disabled = controlsBusy;
        }
        if (moduleInfo) {
            moduleInfo.style.pointerEvents = controlsBusy ? "none" : "";
            moduleInfo.style.opacity = controlsBusy ? "0.85" : "";
            for (const btn of moduleInfo.querySelectorAll(".alexz-mod-picker-action-row .alexz-mod-picker-btn-small")) {
                btn.disabled = busy;
            }
        }
        if (nodeList) {
            nodeList.style.pointerEvents = controlsBusy ? "none" : "";
            nodeList.style.opacity = controlsBusy ? "0.92" : "";
        }

        const processUi = getProcessUi();
        processUi?.setButtonsDisabled?.(busy);
    };

    const setActionBusy = (busy) => {
        if (!shouldContinue()) {
            return;
        }
        actionBusy = Boolean(busy);
        syncBusyUiState();
    };

    const setStartupBusy = (busy) => {
        if (!shouldContinue()) {
            return;
        }
        startupBusy = Boolean(busy);
        syncBusyUiState();
    };

    const setCatalogControlsLoading = (loading) => {
        if (!shouldContinue()) {
            return;
        }
        const busy = Boolean(loading);
        catalogControlsLoading = busy;
        const groupSelect = controls.groupSelect;
        const nodeSelect = controls.nodeSelect;
        if (busy && groupSelect && groupSelect.options.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "Loading groups...";
            groupSelect.appendChild(opt);
        }
        if (busy && nodeSelect && nodeSelect.options.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "Loading modules...";
            nodeSelect.appendChild(opt);
        }
        syncBusyUiState();
    };

    return {
        syncBusyUiState,
        setActionBusy,
        setStartupBusy,
        setCatalogControlsLoading,
        getActionBusy: () => Boolean(actionBusy),
        getStartupBusy: () => Boolean(startupBusy),
        isActionBusy: () => Boolean(actionBusy || startupBusy),
        isCatalogControlsLoading: () => Boolean(catalogControlsLoading),
    };
}
