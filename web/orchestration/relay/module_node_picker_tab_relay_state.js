/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_state.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shared relay-state access helpers for Module Node Picker tab relay.
 *
 * Purpose:
 *   Centralizes read/write operations for the global relay runtime state key
 *   on `window` to keep facade implementation focused on relay behavior.
 */

const TAB_RELAY_STATE_KEY = "__alexz_module_picker_tab_relay_state_v2__";

/**
 * Return current relay state object from global runtime storage.
 */
export function getModuleNodePickerRelayState() {
    return window[TAB_RELAY_STATE_KEY];
}

/**
 * Persist relay state object into global runtime storage.
 */
export function setModuleNodePickerRelayState(state) {
    window[TAB_RELAY_STATE_KEY] = state;
}

/**
 * Clear relay state from global runtime storage.
 */
export function clearModuleNodePickerRelayState() {
    window[TAB_RELAY_STATE_KEY] = null;
}

