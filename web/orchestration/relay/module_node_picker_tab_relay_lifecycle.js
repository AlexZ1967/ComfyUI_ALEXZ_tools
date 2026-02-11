/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_lifecycle.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Relay bind-state lifecycle helpers for Module Node Picker tab relay.
 *
 * Purpose:
 *   Centralizes bind-state construction and deterministic cleanup so facade
 *   bind/unbind paths stay small and consistent.
 */

import { unbindRelayDomEvents } from "./module_node_picker_tab_relay_events.js";

/**
 * Build relay state object from normalized handler/runtime hooks.
 */
export function createRelayBindState({
    bindToken,
    handlers = {},
    stopIntent,
    stopTick,
    dispose,
}) {
    return {
        bindToken,
        relayTimer: 0,
        tickTimer: 0,
        ...handlers,
        stopIntent: typeof stopIntent === "function" ? stopIntent : () => {},
        stopTick: typeof stopTick === "function" ? stopTick : () => {},
        dispose: typeof dispose === "function" ? dispose : () => {},
    };
}

/**
 * Stop timers/listeners/resources for a relay state object.
 */
export function disposeRelayBindState(state) {
    if (!state || typeof window === "undefined" || typeof document === "undefined") {
        return;
    }
    if (state.relayTimer) {
        window.clearTimeout(state.relayTimer);
    }
    if (state.tickTimer) {
        window.clearTimeout(state.tickTimer);
    }
    // Backward-compat cleanup for previous relay state shape.
    if (state.tickInterval) {
        window.clearInterval(state.tickInterval);
    }
    if (typeof state.stopTick === "function") {
        state.stopTick();
    }
    if (typeof state.stopIntent === "function") {
        state.stopIntent();
    }
    unbindRelayDomEvents(state);
    if (typeof state.dispose === "function") {
        state.dispose();
    }
}
