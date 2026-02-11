/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_runtime.js
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
} from "./module_node_picker_tab_relay_helpers.js";
import {
    buildRelayDiagnosticsPayload,
    createRelayDiagnosticsEmitter,
} from "./module_node_picker_tab_relay_diagnostics.js";
import { createRelayDomOwnershipController } from "./module_node_picker_tab_relay_dom_ownership.js";
import {
    RELAY_FOREIGN_TAB_HIDE_MS,
    RELAY_MIN_SYNC_INTERVAL_MS,
    RELAY_REASON_INIT,
    RELAY_REASON_FOREIGN_TAB_CLICK,
    RELAY_REASON_OWN_TAB_CLICK,
    RELAY_REASON_TICK,
    RELAY_REASON_WAIT_FOREIGN_TAB,
    isImmediateRelayReason,
} from "./module_node_picker_tab_relay_constants.js";

/**
 * Create relay runtime with deterministic attach/detach and diagnostics.
 */
export function createModuleNodePickerTabRelayRuntime({ app, root, mountHost, sidebarTabId, onDiag }) {
    let lastClickedTabId = "";
    let pendingForeignTabId = "";
    let pendingForeignTabAt = 0;
    const emitDiagnostics = createRelayDiagnosticsEmitter(onDiag);
    const domOwnership = createRelayDomOwnershipController({ root, mountHost });
    const FOREIGN_TAB_HIDE_MS = RELAY_FOREIGN_TAB_HIDE_MS;
    const MIN_SYNC_INTERVAL_MS = RELAY_MIN_SYNC_INTERVAL_MS;
    let lastSyncAt = 0;
    let pendingSyncReason = "";
    let pendingSyncClickedTabId = "";
    let pendingSyncTimer = 0;
    let lastAppliedVisibilityState = "unknown";

    /**
     * Emit deduplicated diagnostics payload to panel callback.
     */
    const emitDiag = (reason, clickedTabId = "") => {
        emitDiagnostics(buildRelayDiagnosticsPayload({
            app,
            root,
            sidebarTabId,
            reason,
            clickedTabId,
            lastClickedTabId,
        }));
    };

    /**
     * Start temporary foreign-tab protection window to avoid stale ownership flicker.
     */
    const markForeignTabIntent = (tabId) => {
        const normalized = String(tabId || "").trim() || "(unknown-other-tab)";
        pendingForeignTabId = normalized;
        pendingForeignTabAt = Date.now();
        lastClickedTabId = normalized;
        domOwnership.ensureDetached();
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
                domOwnership.ensureAttached();
            }
            root.style.display = "";
            lastAppliedVisibilityState = "shown";
        } else {
            const needDetach = root.isConnected || lastAppliedVisibilityState !== "hidden";
            if (needDetach) {
                domOwnership.ensureDetached();
            }
            lastAppliedVisibilityState = "hidden";
        }
        const effectiveReason = foreignIntentActive && reason !== RELAY_REASON_OWN_TAB_CLICK
            ? RELAY_REASON_WAIT_FOREIGN_TAB
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
        const normalizedReason = String(reason || RELAY_REASON_TICK);
        const normalizedTabId = String(clickedTabId || "");
        const isImmediateReason = isImmediateRelayReason(normalizedReason);
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
            const queuedReason = pendingSyncReason || RELAY_REASON_TICK;
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
            syncVisibility(RELAY_REASON_OWN_TAB_CLICK, String(tabId || sidebarTabId));
        },

        /**
         * Mark click on foreign/unknown tab and keep picker detached.
         */
        onForeignTabClick(tabId) {
            markForeignTabIntent(tabId);
            syncVisibility(RELAY_REASON_FOREIGN_TAB_CLICK, String(tabId || ""));
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
