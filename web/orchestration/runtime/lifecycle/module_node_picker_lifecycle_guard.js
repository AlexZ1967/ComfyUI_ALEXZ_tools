/**
 * Module: web/orchestration/runtime/lifecycle/module_node_picker_lifecycle_guard.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Shared lifecycle guard helpers for Module Node Picker async flows.
 *
 * Purpose:
 *   Provides a single should-continue check used by actions/update orchestration
 *   to stop stale async work after picker dispose.
 */

/**
 * Return true while current picker/action context is still valid.
 */
export function shouldContinueContext(context) {
    const fn = context?.shouldContinue;
    if (typeof fn !== "function") {
        return true;
    }
    try {
        return fn() !== false;
    } catch (_err) {
        return false;
    }
}
