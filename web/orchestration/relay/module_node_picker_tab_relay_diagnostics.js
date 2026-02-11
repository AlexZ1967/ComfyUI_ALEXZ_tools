/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_diagnostics.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Diagnostics payload helpers for Module Node Picker tab relay runtime.
 *
 * Purpose:
 *   Centralizes diagnostics payload shape and duplicate-event suppression so
 *   runtime visibility logic stays focused on ownership decisions.
 */

import {
    getActiveSidebarTabId,
    isOwnButtonSelected,
    getContainerState,
} from "./module_node_picker_tab_relay_helpers.js";

/**
 * Build diagnostics payload for current relay/runtime state.
 */
export function buildRelayDiagnosticsPayload({
    app,
    root,
    sidebarTabId,
    reason,
    clickedTabId,
    lastClickedTabId,
}) {
    const ownSelected = isOwnButtonSelected(sidebarTabId);
    const containerState = getContainerState(root);
    return {
        reason,
        activeTabId: getActiveSidebarTabId(app) || "n/a",
        lastClickedTabId: clickedTabId || lastClickedTabId || "n/a",
        ownBtnFound: ownSelected !== null,
        ownBtnSelected: ownSelected,
        rootDisplay: root?.style?.display || "",
        childNodesCount: containerState.childCount,
        childNodesShort: containerState.childShort,
    };
}

/**
 * Create deduplicated diagnostics emitter.
 */
export function createRelayDiagnosticsEmitter(onDiag) {
    let lastDiagSig = "";
    return (diag) => {
        const sig = JSON.stringify(diag || {});
        if (sig === lastDiagSig) {
            return;
        }
        lastDiagSig = sig;
        onDiag?.(diag);
    };
}
