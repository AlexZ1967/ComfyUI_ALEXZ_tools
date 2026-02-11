/**
 * Module: web/orchestration/relay/module_node_picker_tab_relay_constants.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shared constants for Module Node Picker tab relay.
 *
 * Purpose:
 *   Centralizes relay reason labels and timing thresholds to keep relay
 *   runtime/intent/tick modules synchronized and easier to maintain.
 */

// Relay reason labels used in diagnostics and sync orchestration.
export const RELAY_REASON_INIT = "relay_init";
export const RELAY_REASON_TICK = "relay_tick";
export const RELAY_REASON_OWN_TAB_CLICK = "relay_own_tab_click";
export const RELAY_REASON_FOREIGN_TAB_CLICK = "relay_foreign_tab_click";
export const RELAY_REASON_NATIVE_OK = "relay_native_ok";
export const RELAY_REASON_PENDING_SWITCH = "relay_pending_switch";
export const RELAY_REASON_WAIT_FOREIGN_TAB = "relay_wait_foreign_tab";
export const RELAY_REASON_KEYUP = "relay_keyup";
export const RELAY_REASON_VISIBILITY = "relay_visibility";
export const RELAY_REASON_PAGESHOW = "relay_pageshow";

// Relay timing controls.
export const RELAY_FOREIGN_TAB_HIDE_MS = 1600;
export const RELAY_MIN_SYNC_INTERVAL_MS = 48;
export const RELAY_FOREIGN_CORRECTION_DELAY_MS = 60;
export const RELAY_TICK_FAST_MS = 500;
export const RELAY_TICK_IDLE_MS = 900;
export const RELAY_PASSIVE_TICK_BUDGET_LIMIT = 6;

const IMMEDIATE_RELAY_REASONS = new Set([
    RELAY_REASON_OWN_TAB_CLICK,
    RELAY_REASON_FOREIGN_TAB_CLICK,
    RELAY_REASON_NATIVE_OK,
    RELAY_REASON_PENDING_SWITCH,
    RELAY_REASON_INIT,
]);

/**
 * Return true when relay reason should bypass debounce delay.
 */
export function isImmediateRelayReason(reason) {
    return IMMEDIATE_RELAY_REASONS.has(String(reason || ""));
}
