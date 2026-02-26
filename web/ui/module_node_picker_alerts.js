/**
 * Module: web/ui/module_node_picker_alerts.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Status-card render helpers for ComfyUI and Custom Nodes update sections.
 *
 * Purpose:
 *   Keeps alert-card rendering separate from picker orchestration and backend calls.
 */

/**
 * Render ComfyUI status card for monitoring-only mode.
 */
export function renderComfyAlertCard(context) {
    const info = context?.info || null;
    const comfyMode = String(context?.comfyMode || "releases");
    const fmtDate = typeof context?.fmtDate === "function"
        ? context.fmtDate
        : (value) => String(value || "n/a");
    const comfyAlert = context?.comfyAlert;
    const comfyAlertText = context?.comfyAlertText;

    if (!comfyAlert || !comfyAlertText) {
        return;
    }

    const behind = Number(info?.behind);
    const status = String(info?.update_status || "unknown");
    const mode = String(info?.check_mode || comfyMode || "releases");
    const branch = String(info?.branch || "unknown");
    const local = String(info?.installed_commit_short || "unknown");
    const remote = String(info?.remote_commit_short || "unknown");
    const releaseTag = String(info?.release_tag || "").trim();
    const canUpdate = status === "can_update" && (!Number.isFinite(behind) || behind > 0);
    const requirementsPending = Boolean(info?.requirements_update_pending);
    const requirementsPendingAt = info?.requirements_pending_updated_at
        ? ` (${fmtDate(info.requirements_pending_updated_at)})`
        : "";

    comfyAlert.classList.remove(
        "alexz-mod-picker-status-card--warn",
        "alexz-mod-picker-status-card--ok",
        "alexz-mod-picker-status-card--neutral"
    );
    comfyAlert.style.display = "block";

    if (canUpdate) {
        comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
        if (mode === "releases" && releaseTag) {
            comfyAlertText.textContent = `ComfyUI requires update (releases): release=${releaseTag}, local=${local}, remote=${remote}.`;
        } else if (Number.isFinite(behind) && behind > 0) {
            comfyAlertText.textContent = `ComfyUI requires update (commits): branch=${branch}, behind=${behind}, local=${local}, remote=${remote}.`;
        } else {
            comfyAlertText.textContent = `ComfyUI requires update: mode=${mode}, branch=${branch}, local=${local}, remote=${remote}.`;
        }
        if (requirementsPending) {
            comfyAlertText.textContent += ` requirements.txt install is pending${requirementsPendingAt}.`;
        }
        return;
    }

    if (requirementsPending) {
        comfyAlert.classList.add("alexz-mod-picker-status-card--warn");
        comfyAlertText.textContent = `ComfyUI requirements.txt install is pending${requirementsPendingAt}.`;
        return;
    }

    comfyAlert.classList.add("alexz-mod-picker-status-card--neutral");
    if (Boolean(info?.updated_between_runs)) {
        const prev = String(info?.startup_prev_commit_short || "unknown");
        const next = String(info?.startup_new_commit_short || "unknown");
        const at = info?.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
        comfyAlertText.textContent = `ComfyUI updated between runs: ${prev} -> ${next}${at}. No updates required.`;
    } else {
        comfyAlertText.textContent = `ComfyUI is up to date (${mode} check).`;
    }
}

/**
 * Render Custom Nodes global status card for monitoring-only mode.
 */
export function renderCustomAlertCard(context) {
    const customModulesNeedUpdate = Number(context?.customModulesNeedUpdate || 0);
    const customModulesUnknownUpdate = Number(context?.customModulesUnknownUpdate || 0);
    const customStatusChecked = Boolean(context?.customStatusChecked);
    const customAlert = context?.customAlert;
    const customAlertText = context?.customAlertText;

    if (!customAlert || !customAlertText) {
        return;
    }

    if (!customStatusChecked) {
        customAlert.style.display = "none";
        return;
    }
    customAlert.classList.remove(
        "alexz-mod-picker-status-card--warn",
        "alexz-mod-picker-status-card--ok",
        "alexz-mod-picker-status-card--neutral"
    );
    customAlert.style.display = "block";
    if (customModulesNeedUpdate > 0) {
        customAlert.classList.add("alexz-mod-picker-status-card--warn");
        if (customModulesUnknownUpdate > 0) {
            customAlertText.textContent =
                `${customModulesNeedUpdate} custom modules require update, ` +
                `${customModulesUnknownUpdate} modules could not be checked.`;
        } else {
            customAlertText.textContent = `${customModulesNeedUpdate} custom modules require update.`;
        }
        return;
    }
    if (customModulesUnknownUpdate > 0) {
        customAlert.classList.add("alexz-mod-picker-status-card--warn");
        customAlertText.textContent =
            `${customModulesUnknownUpdate} custom modules could not be checked for updates (missing git remote/upstream).`;
        return;
    }
    customAlert.classList.add("alexz-mod-picker-status-card--neutral");
    customAlertText.textContent = "Custom Nodes: no updates required.";
}
