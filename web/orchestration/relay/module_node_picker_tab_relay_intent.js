/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_intent.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Foreign-tab intent and event-handling controller for tab relay.
 *
 * Purpose:
 *   Encapsulates sidebar tab interaction detection and relay correction timer
 *   logic, keeping main relay binding file focused on listener wiring.
 */

import {
    getActiveSidebarTabId,
    resolveSidebarButtonFromEvent,
    isOwnButtonSelected,
    inferTabIdFromButton,
} from "./module_node_picker_tab_relay_helpers.js";
import {
    RELAY_FOREIGN_CORRECTION_DELAY_MS,
    RELAY_REASON_KEYUP,
    RELAY_REASON_NATIVE_OK,
    RELAY_REASON_PAGESHOW,
    RELAY_REASON_PENDING_SWITCH,
    RELAY_REASON_VISIBILITY,
} from "./module_node_picker_tab_relay_constants.js";

/**
 * Create intent controller for tab interactions and relay correction scheduling.
 */
export function createModuleNodePickerRelayIntentController(context = {}) {
    const app = context?.app;
    const root = context?.root;
    const sidebarTabId = String(context?.sidebarTabId || "");
    const relayRuntime = context?.relayRuntime;
    const isCurrentBinding = typeof context?.isCurrentBinding === "function"
        ? context.isCurrentBinding
        : () => false;
    const getLiveState = typeof context?.getLiveState === "function"
        ? context.getLiveState
        : () => null;

    let relayTimer = 0;

    const setRelayTimer = (timer) => {
        relayTimer = timer;
        const liveState = getLiveState();
        if (liveState) {
            liveState.relayTimer = timer;
        }
    };

    const clearRelayTimer = () => {
        if (!relayTimer) {
            return;
        }
        window.clearTimeout(relayTimer);
        setRelayTimer(0);
    };

    const processTabId = (tabId) => {
        if (tabId === sidebarTabId) {
            relayRuntime?.onOwnTabClick?.(sidebarTabId);
            return;
        }
        relayRuntime?.onForeignTabClick?.(tabId);
        if (isOwnButtonSelected(sidebarTabId) !== true) {
            return;
        }
        clearRelayTimer();
        setRelayTimer(window.setTimeout(() => {
            if (!isCurrentBinding()) {
                return;
            }
            setRelayTimer(0);
            const activeTabId = getActiveSidebarTabId(app);
            const ownSelected = isOwnButtonSelected(sidebarTabId);
            if (activeTabId === tabId || ownSelected === false) {
                relayRuntime?.clearForeignIntent?.();
                relayRuntime?.syncVisibility?.(RELAY_REASON_NATIVE_OK, tabId);
                return;
            }
            // Do not force tab activation from relay. Only re-evaluate visibility.
            relayRuntime?.syncVisibility?.(RELAY_REASON_PENDING_SWITCH, tabId);
        }, RELAY_FOREIGN_CORRECTION_DELAY_MS));
    };

    const handleEvent = (event) => {
        if (!isCurrentBinding()) {
            return;
        }
        // React only to primary-button pointer/mouse events.
        if (event?.type === "pointerdown" || event?.type === "mousedown") {
            const button = Number(event?.button);
            if (Number.isFinite(button) && button !== 0) {
                return;
            }
        }
        const direct = event?.target;
        // Ignore any interaction that originates inside picker root,
        // including text-node targets emitted by some browser/event paths.
        if (direct instanceof Node && root?.contains?.(direct)) {
            return;
        }
        if (typeof event?.composedPath === "function" && event.composedPath().includes(root)) {
            return;
        }
        const button = resolveSidebarButtonFromEvent(event);
        if (!button) {
            return;
        }
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId) {
            return;
        }
        processTabId(tabId);
    };

    const onKeyUp = (event) => {
        if (!isCurrentBinding()) {
            return;
        }
        const key = String(event?.key || "");
        const relevantKeys = [
            "Enter",
            " ",
            "Spacebar",
            "ArrowLeft",
            "ArrowRight",
            "ArrowUp",
            "ArrowDown",
            "Home",
            "End",
            "Tab",
        ];
        if (key && !relevantKeys.includes(key)) {
            return;
        }
        const target = event?.target;
        if (target instanceof Node && root?.contains?.(target)) {
            return;
        }
        if (target instanceof Element && target.closest("input, textarea, [contenteditable]")) {
            return;
        }
        relayRuntime?.syncVisibility?.(RELAY_REASON_KEYUP);
    };

    const onVisibilityChange = () => {
        if (!isCurrentBinding()) {
            return;
        }
        relayRuntime?.syncVisibility?.(RELAY_REASON_VISIBILITY);
    };

    const onPageShow = () => {
        if (!isCurrentBinding()) {
            return;
        }
        relayRuntime?.syncVisibility?.(RELAY_REASON_PAGESHOW);
    };

    return {
        handleEvent,
        onKeyUp,
        onVisibilityChange,
        onPageShow,
        dispose() {
            clearRelayTimer();
        },
    };
}
