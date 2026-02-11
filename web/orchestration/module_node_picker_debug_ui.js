/**
 * Module: web/orchestration/module_node_picker_debug_ui.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Debug/diagnostics UI controller for Module Node Picker.
 *
 * Purpose:
 *   Encapsulates debug toggle state, diagnostics rendering, clipboard copy,
 *   and cleanup of listeners/subscriptions across picker re-renders.
 */

/**
 * Create debug UI controller for picker diagnostics panel.
 */
export function createModuleNodePickerDebugUi(context) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const windowObj = context?.windowObj || window;
    const debugStateKey = String(context?.debugStateKey || "");
    const pickerStore = context?.pickerStore || null;
    const diagnosticsLogger = context?.diagnosticsLogger || null;
    const debugToggle = context?.debugToggle || null;
    const debugCard = context?.debugCard || null;
    const debugCopyBtn = context?.debugCopyBtn || null;
    const diagnostics = context?.diagnostics || null;
    const onCopyStatus = typeof context?.onCopyStatus === "function"
        ? context.onCopyStatus
        : () => {};

    let debugEnabled = Boolean(pickerStore?.get?.("debugEnabled"));
    const unsubscribeDebug = pickerStore?.subscribe?.("debugEnabled", (value) => {
        debugEnabled = Boolean(value);
        applyDebugUiState();
    }) || (() => {});

    function applyDebugUiState() {
        windowObj[debugStateKey] = Boolean(debugEnabled);
        diagnosticsLogger?.setDebugEnabled?.(Boolean(debugEnabled));
        if (debugCard) {
            debugCard.hidden = !debugEnabled;
            debugCard.style.display = debugEnabled ? "block" : "none";
        }
        if (debugToggle) {
            debugToggle.textContent = debugEnabled ? "Debug: ON" : "Debug";
        }
    }

    const onDebugToggleClick = () => {
        pickerStore?.set?.({ debugEnabled: !Boolean(pickerStore?.get?.("debugEnabled")) });
    };
    const onDebugCopyClick = async () => {
        try {
            await navigator.clipboard.writeText(diagnostics?.textContent || "");
            onCopyStatus("Debug diagnostics copied to clipboard.");
        } catch (_err) {
            onCopyStatus("Failed to copy debug diagnostics.");
        }
    };

    debugToggle?.addEventListener?.("click", onDebugToggleClick);
    debugCopyBtn?.addEventListener?.("click", onDebugCopyClick);
    applyDebugUiState();

    function setDiagnosticText(diag) {
        if (!shouldContinue()) {
            return;
        }
        if (!diagnostics) {
            return;
        }
        const lines = [
            `diag.ts=${new Date().toLocaleTimeString()}`,
            `diag.reason=${diag?.reason || "unknown"}`,
            `diag.active_tab=${diag?.activeTabId || "n/a"}`,
            `diag.last_clicked_tab=${diag?.lastClickedTabId || "n/a"}`,
            `diag.own_btn_found=${diag?.ownBtnFound ? "yes" : "no"}`,
            `diag.own_btn_selected=${diag?.ownBtnSelected === null ? "n/a" : (diag?.ownBtnSelected ? "yes" : "no")}`,
            `diag.root_display=${diag?.rootDisplay || "n/a"}`,
            `diag.child_nodes=${Number(diag?.childNodesCount || 0)}`,
            `diag.child_nodes_short=${diag?.childNodesShort || "n/a"}`,
        ];
        diagnostics.textContent = lines.join("\n");
    }

    function dispose() {
        try {
            unsubscribeDebug?.();
        } catch (_err) {
            // Ignore stale store-unsubscribe errors.
        }
        try {
            debugToggle?.removeEventListener?.("click", onDebugToggleClick);
        } catch (_err) {
            // Ignore stale listener-cleanup errors.
        }
        try {
            debugCopyBtn?.removeEventListener?.("click", onDebugCopyClick);
        } catch (_err) {
            // Ignore stale listener-cleanup errors.
        }
    }

    return {
        setDiagnosticText,
        dispose,
    };
}
