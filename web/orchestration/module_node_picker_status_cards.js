/**
 * Module: web/orchestration/module_node_picker_status_cards.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   Status-card controller for ComfyUI and Custom Nodes sections.
 *
 * Purpose:
 *   Encapsulates status card rendering and checked-state persistence so picker
 *   orchestration can consume a stable API instead of managing card state inline.
 */

import {
    renderComfyAlertCard,
    renderCustomAlertCard,
} from "../ui/module_node_picker_alerts.js";

/**
 * Create status-card controller for picker top cards.
 */
export function createModuleNodePickerStatusCards(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const getComfyMode = typeof context?.getComfyMode === "function"
        ? context.getComfyMode
        : () => "releases";
    const getActionBusy = typeof context?.getActionBusy === "function"
        ? context.getActionBusy
        : () => false;
    const fmtDate = context?.fmtDate;
    const comfyAlert = context?.comfyAlert || null;
    const comfyAlertText = context?.comfyAlertText || null;
    const comfyUpdateBtn = context?.comfyUpdateBtn || null;
    const comfyInstallReqBtn = context?.comfyInstallReqBtn || null;
    const customAlert = context?.customAlert || null;
    const customAlertText = context?.customAlertText || null;
    const updateAllBtn = context?.updateAllBtn || null;
    const getCustomModulesNeedUpdate = typeof context?.getCustomModulesNeedUpdate === "function"
        ? context.getCustomModulesNeedUpdate
        : () => 0;
    const saveCustomStatusChecked = typeof context?.saveCustomStatusChecked === "function"
        ? context.saveCustomStatusChecked
        : () => {};
    const saveComfyStatusChecked = typeof context?.saveComfyStatusChecked === "function"
        ? context.saveComfyStatusChecked
        : () => {};
    const saveComfyInfoSnapshot = typeof context?.saveComfyInfoSnapshot === "function"
        ? context.saveComfyInfoSnapshot
        : () => {};

    let customStatusChecked = Boolean(context?.initialCustomStatusChecked);
    let comfyStatusChecked = Boolean(context?.initialComfyStatusChecked);

    const renderComfyAlert = (info) => {
        if (!shouldContinue()) {
            return;
        }
        if (info && typeof info === "object") {
            comfyStatusChecked = true;
            saveComfyStatusChecked(true);
            saveComfyInfoSnapshot(info);
        }
        renderComfyAlertCard({
            info,
            comfyMode: getComfyMode(),
            actionBusy: getActionBusy(),
            fmtDate,
            comfyAlert,
            comfyAlertText,
            comfyUpdateBtn,
            comfyInstallReqBtn,
        });
    };

    const renderCustomAlert = () => {
        if (!shouldContinue()) {
            return;
        }
        renderCustomAlertCard({
            customModulesNeedUpdate: Number(getCustomModulesNeedUpdate() || 0),
            customStatusChecked,
            actionBusy: getActionBusy(),
            customAlert,
            customAlertText,
            updateAllBtn,
        });
    };

    const setCustomStatusChecked = (checked) => {
        customStatusChecked = Boolean(checked);
        saveCustomStatusChecked(customStatusChecked);
        renderCustomAlert();
    };

    const setComfyStatusChecked = (checked) => {
        comfyStatusChecked = Boolean(checked);
        saveComfyStatusChecked(comfyStatusChecked);
    };

    const syncUpdateAllButton = () => {
        renderCustomAlert();
    };

    return {
        renderComfyAlert,
        renderCustomAlert,
        setCustomStatusChecked,
        setComfyStatusChecked,
        syncUpdateAllButton,
        getCustomStatusChecked: () => customStatusChecked,
        getComfyStatusChecked: () => comfyStatusChecked,
    };
}
