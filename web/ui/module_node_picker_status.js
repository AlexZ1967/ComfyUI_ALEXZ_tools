/**
 * Module: web/ui/module_node_picker_status.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Status text formatters for Module Node Picker long-running operations.
 *
 * Purpose:
 *   Converts backend refresh/update payloads into consistent one-line UI states.
 */

/**
 * Convert backend refresh status payload into a one-line progress message.
 */
export function formatRefreshLine(refresh) {
    const phase = String(refresh?.phase || "");
    const current = Number(refresh?.current || 0);
    const total = Number(refresh?.total || 0);
    const remaining = Number(refresh?.remaining || 0);
    const modulesNeedUpdate = Number(refresh?.modules_need_update || 0);
    const modulesUnknownUpdate = Number(refresh?.modules_unknown_update || 0);
    const unknownUpdateModules = Array.isArray(refresh?.unknown_update_modules)
        ? refresh.unknown_update_modules.map((name) => String(name || "").trim()).filter(Boolean)
        : [];
    const moduleName = String(refresh?.module || "");
    const error = String(refresh?.error || "");

    if (phase === "sync") {
        if (total > 0) {
            const modulePart = moduleName ? ` (${moduleName})` : "";
            return { text: `Refreshing Custom Nodes: ${current}/${total}, remaining ${remaining}${modulePart}`, tone: "neutral" };
        }
        return { text: "Refreshing Custom Nodes: preparing...", tone: "neutral" };
    }
    if (phase === "snapshots") {
        return { text: "Refreshing Custom Nodes: recomputing snapshots...", tone: "neutral" };
    }
    if (phase === "done") {
        const count = Number.isFinite(modulesNeedUpdate) ? Math.max(0, modulesNeedUpdate) : 0;
        const unknown = Number.isFinite(modulesUnknownUpdate) ? Math.max(0, modulesUnknownUpdate) : 0;
        const unknownPreview = unknownUpdateModules.slice(0, 3).join(", ");
        const unknownTail = unknownUpdateModules.length > 3
            ? `, +${unknownUpdateModules.length - 3} more`
            : "";
        const unknownNamesPart = unknownPreview
            ? ` (${unknownPreview}${unknownTail})`
            : "";
        if (count > 0) {
            if (unknown > 0) {
                return {
                    text: `${count} custom modules require update, ${unknown} could not be checked${unknownNamesPart}`,
                    tone: "warn",
                };
            }
            return { text: `${count} custom modules require update`, tone: "warn" };
        }
        if (unknown > 0) {
            return {
                text: `${unknown} custom modules could not be checked${unknownNamesPart}`,
                tone: "warn",
            };
        }
        return { text: "Custom Nodes: no updates required", tone: "ok" };
    }
    if (phase === "error") {
        return { text: `Custom Nodes refresh failed${error ? ` (${error})` : ""}.`, tone: "warn" };
    }
    return { text: "Refreshing Custom Nodes: starting...", tone: "neutral" };
}

/**
 * Convert module-update status payload into a one-line progress/result message.
 */
export function formatUpdateLine(update) {
    const scope = String(update?.scope || "");
    const phase = String(update?.phase || "");
    const current = Number(update?.current || 0);
    const total = Number(update?.total || 0);
    const remaining = Number(update?.remaining || 0);
    const moduleName = String(update?.module || "");
    const error = String(update?.error || "");
    const updated = Number(update?.updated || 0);
    const failed = Number(update?.failed || 0);
    const requirementsChanged = Boolean(update?.requirements_changed);
    const reqList = Array.isArray(update?.requirements_modules) ? update.requirements_modules : [];

    if (phase === "update") {
        const modulePart = moduleName ? ` (${moduleName})` : "";
        if (total > 0) {
            return { text: `Updating modules: ${current}/${total}, remaining ${remaining}${modulePart}`, tone: "neutral" };
        }
        return { text: "Updating modules: starting...", tone: "neutral" };
    }
    if (phase === "done") {
        if (scope === "comfyui") {
            if (failed > 0) {
                return { text: "ComfyUI update finished with errors.", tone: "warn" };
            }
            if (updated > 0 && requirementsChanged) {
                return { text: "ComfyUI updated. requirements.txt changed.", tone: "warn" };
            }
            if (updated > 0) {
                return { text: "ComfyUI updated.", tone: "ok" };
            }
            return { text: "ComfyUI already up to date.", tone: "ok" };
        }
        if (total <= 0) {
            return { text: "No updates found.", tone: "ok" };
        }
        if (failed > 0) {
            return { text: `Update finished: updated=${updated}, failed=${failed}.`, tone: "warn" };
        }
        if (reqList.length > 0) {
            return { text: `Update finished: ${updated} module(s) updated, requirements changed.`, tone: "warn" };
        }
        return { text: `Update finished: ${updated} module(s) updated.`, tone: "ok" };
    }
    if (phase === "error") {
        return { text: `Update failed${error ? ` (${error})` : ""}.`, tone: "warn" };
    }
    return { text: "Preparing update...", tone: "neutral" };
}
