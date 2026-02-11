/**
 * Module: web/orchestration/module_node_picker_error_utils.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Error classification helpers for Module Node Picker async flows.
 *
 * Purpose:
 *   Provides stable cancellation/abort detection so orchestration code can
 *   suppress non-actionable warnings for intentionally canceled requests.
 */

/**
 * Return true when error indicates an intentionally canceled/aborted request.
 */
export function isCanceledRequestError(err) {
    const name = String(err?.name || "").trim().toLowerCase();
    const message = String(err?.message || err || "").trim().toLowerCase();
    if (!name && !message) {
        return false;
    }
    if (name === "aborterror") {
        return true;
    }
    return (
        message.includes("request canceled")
        || message.includes("request cancelled")
        || message.includes("request aborted")
        || message.includes("aborterror")
    );
}

