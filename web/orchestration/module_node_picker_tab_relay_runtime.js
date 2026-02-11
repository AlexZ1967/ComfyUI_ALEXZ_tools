/**
 * Module: web/orchestration/module_node_picker_tab_relay_runtime.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Runtime/state controller for Module Node Picker tab relay.
 *
 * Purpose:
 *   Encapsulates visibility ownership rules, foreign-tab intent handling,
 *   and deduplicated diagnostics emission used by tab relay wiring.
 */

import {
    getActiveSidebarTabId,
    isOwnButtonSelected,
    getContainerState,
} from "./module_node_picker_tab_relay_helpers.js";

/**
 * Create relay runtime with deterministic attach/detach and diagnostics.
 */
export function createModuleNodePickerTabRelayRuntime({ app, root, sidebarTabId, onDiag }) {
    let lastClickedTabId = "";
    let pendingForeignTabId = "";
    let pendingForeignTabAt = 0;
    let lastDiagSig = "";
    const homeContainer = root.parentElement instanceof Element ? root.parentElement : null;
    const FOREIGN_TAB_HIDE_MS = 1600;
    const MIN_SYNC_INTERVAL_MS = 48;
    let lastSyncAt = 0;
    let pendingSyncReason = "";
    let pendingSyncClickedTabId = "";
    let pendingSyncTimer = 0;
    let lastAppliedVisibilityState = "unknown";

    /**
     * Emit deduplicated diagnostics payload to panel callback.
     */
    const emitDiag = (reason, clickedTabId = "") => {
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const containerState = getContainerState(root);
        const diag = {
            reason,
            activeTabId: getActiveSidebarTabId(app) || "n/a",
            lastClickedTabId: clickedTabId || lastClickedTabId || "n/a",
            ownBtnFound: ownSelected !== null,
            ownBtnSelected: ownSelected,
            rootDisplay: root.style.display || "",
            childNodesCount: containerState.childCount,
            childNodesShort: containerState.childShort,
        };
        const sig = JSON.stringify(diag);
        if (sig === lastDiagSig) {
            return;
        }
        lastDiagSig = sig;
        onDiag?.(diag);
    };

    /**
     * Re-attach picker root into home container when needed.
     */
    const ensureRootAttached = () => {
        if (root.isConnected) {
            return true;
        }
        if (homeContainer && homeContainer.isConnected) {
            homeContainer.appendChild(root);
            return true;
        }
        return false;
    };

    /**
     * Detach picker root from DOM when another sidebar tab should own panel area.
     */
    const ensureRootDetached = () => {
        if (!root.isConnected) {
            return true;
        }
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        return true;
    };

    /**
     * Start temporary foreign-tab protection window to avoid stale ownership flicker.
     */
    const markForeignTabIntent = (tabId) => {
        const normalized = String(tabId || "").trim() || "(unknown-other-tab)";
        pendingForeignTabId = normalized;
        pendingForeignTabAt = Date.now();
        lastClickedTabId = normalized;
        ensureRootDetached();
    };

    /**
     * Reset temporary foreign-tab protection window.
     */
    const clearForeignTabIntent = () => {
        pendingForeignTabId = "";
        pendingForeignTabAt = 0;
    };

    /**
     * Return true while we intentionally keep root detached after foreign click.
     */
    const isForeignIntentActive = () => {
        if (!pendingForeignTabId) {
            return false;
        }
        if (Date.now() - pendingForeignTabAt > FOREIGN_TAB_HIDE_MS) {
            clearForeignTabIntent();
            return false;
        }
        return true;
    };

    /**
     * Synchronize picker root visibility/attachment with current sidebar state.
     */
    const syncVisibilityNow = (reason, clickedTabId = "") => {
        const activeTabId = getActiveSidebarTabId(app);
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const switchedAway = (activeTabId && activeTabId !== sidebarTabId) || ownSelected === false;
        if (switchedAway) {
            clearForeignTabIntent();
        }
        const foreignIntentActive = isForeignIntentActive();
        let shouldShow = ownSelected === true || activeTabId === sidebarTabId;
        if (shouldShow && foreignIntentActive) {
            shouldShow = false;
        }
        if (shouldShow) {
            const needAttach = !root.isConnected || lastAppliedVisibilityState !== "shown";
            if (needAttach) {
                ensureRootAttached();
            }
            root.style.display = "";
            lastAppliedVisibilityState = "shown";
        } else {
            const needDetach = root.isConnected || lastAppliedVisibilityState !== "hidden";
            if (needDetach) {
                ensureRootDetached();
            }
            lastAppliedVisibilityState = "hidden";
        }
        const effectiveReason = foreignIntentActive && reason !== "relay_own_tab_click"
            ? "relay_wait_foreign_tab"
            : reason;
        emitDiag(effectiveReason, clickedTabId || pendingForeignTabId);
        lastSyncAt = Date.now();
    };

    /**
     * Schedule visibility sync with short debounce to reduce event storms.
     */
    const syncVisibility = (reason, clickedTabId = "") => {
        const now = Date.now();
        const elapsed = now - lastSyncAt;
        const normalizedReason = String(reason || "relay_tick");
        const normalizedTabId = String(clickedTabId || "");
        const isImmediateReason =
            normalizedReason === "relay_own_tab_click"
            || normalizedReason === "relay_foreign_tab_click"
            || normalizedReason === "relay_unknown_tab_click"
            || normalizedReason === "relay_native_ok"
            || normalizedReason === "relay_pending_switch"
            || normalizedReason === "relay_init";
        if (isImmediateReason || elapsed >= MIN_SYNC_INTERVAL_MS) {
            if (pendingSyncTimer) {
                window.clearTimeout(pendingSyncTimer);
                pendingSyncTimer = 0;
            }
            pendingSyncReason = "";
            pendingSyncClickedTabId = "";
            syncVisibilityNow(normalizedReason, normalizedTabId);
            return;
        }
        pendingSyncReason = normalizedReason;
        pendingSyncClickedTabId = normalizedTabId;
        if (pendingSyncTimer) {
            return;
        }
        const delay = Math.max(0, MIN_SYNC_INTERVAL_MS - elapsed);
        pendingSyncTimer = window.setTimeout(() => {
            pendingSyncTimer = 0;
            const queuedReason = pendingSyncReason || "relay_tick";
            const queuedTabId = pendingSyncClickedTabId || "";
            pendingSyncReason = "";
            pendingSyncClickedTabId = "";
            syncVisibilityNow(queuedReason, queuedTabId);
        }, delay);
    };

    return {
        /**
         * Mark click on own tab and re-sync immediately.
         */
        onOwnTabClick(tabId) {
            clearForeignTabIntent();
            lastClickedTabId = String(tabId || sidebarTabId);
            syncVisibility("relay_own_tab_click", String(tabId || sidebarTabId));
        },

        /**
         * Mark click on foreign/unknown tab and keep picker detached.
         */
        onForeignTabClick(tabId) {
            markForeignTabIntent(tabId);
            syncVisibility("relay_foreign_tab_click", String(tabId || ""));
        },

        /**
         * Mark unknown foreign click source and emit unknown-tab diagnostic reason.
         */
        onUnknownForeignTabClick(tabId) {
            markForeignTabIntent(tabId);
            syncVisibility("relay_unknown_tab_click", String(tabId || ""));
        },

        /**
         * Keep temporary pending foreign tab id for delayed re-check.
         */
        getPendingForeignTabId() {
            return String(pendingForeignTabId || "");
        },

        /**
         * Return true when temporary foreign-tab intent window is still active.
         */
        hasPendingForeignIntent() {
            return isForeignIntentActive();
        },

        /**
         * Clear foreign intent when native tab switch is confirmed.
         */
        clearForeignIntent() {
            clearForeignTabIntent();
        },

        /**
         * Stop internal debounce timer when relay is unbound.
         */
        dispose() {
            if (pendingSyncTimer) {
                window.clearTimeout(pendingSyncTimer);
                pendingSyncTimer = 0;
            }
        },

        /**
         * Run visibility synchronization for event/timer reason.
         */
        syncVisibility(reason, clickedTabId = "") {
            syncVisibility(reason, clickedTabId);
        },
    };
}
