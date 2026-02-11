/**
 * Module: web/ui/module_node_picker_formatters.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Formatting helpers for Module Node Picker UI.
 *
 * Purpose:
 *   Keeps pure display/format logic separate from interaction and rendering flow.
 */

/**
 * Format ISO timestamp for local UI display.
 */
export function fmtDate(iso) {
    if (!iso) {
        return "n/a";
    }
    try {
        return new Date(iso).toLocaleString();
    } catch (_err) {
        return String(iso);
    }
}

/**
 * Derive module badge flags from detailed module info payload.
 */
export function moduleBadgesFromInfo(info) {
    const behind = Number(info?.git_behind);
    const status = String(info?.update_status || "");
    return {
        updatedBetweenRuns: Boolean(info?.updated_between_runs),
        hasRemoteUpdate: (Number.isFinite(behind) && behind > 0) || status === "can_update",
        hasUnknownUpdate: status === "unknown",
    };
}

/**
 * Derive module badge flags from lightweight catalog module entry.
 */
export function moduleBadgesFromModuleEntry(entry) {
    const status = String(entry?.update_status || "");
    return {
        updatedBetweenRuns: Boolean(entry?.updated_between_runs) || Boolean(entry?.new_module_between_runs),
        hasRemoteUpdate: Boolean(entry?.update_available),
        hasUnknownUpdate: status === "unknown",
    };
}

/**
 * Build module option text with badges and node count.
 */
export function formatModuleOption(moduleName, count, badges, marks = {}) {
    const updatedMark = String(marks.updatedMark || "✅");
    const remoteUpdateMark = String(marks.remoteUpdateMark || "🟥");
    const unknownUpdateMark = String(marks.unknownUpdateMark || "🟨");
    const markItems = [];
    if (badges?.updatedBetweenRuns) {
        markItems.push(updatedMark);
    }
    if (badges?.hasRemoteUpdate) {
        markItems.push(remoteUpdateMark);
    }
    if (badges?.hasUnknownUpdate) {
        markItems.push(unknownUpdateMark);
    }
    const prefix = markItems.length ? `${markItems.join(" ")} ` : "";
    return `${prefix}${moduleName} (${count})`;
}
