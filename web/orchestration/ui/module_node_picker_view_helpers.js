/**
 * Module: web/orchestration/ui/module_node_picker_view_helpers.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   View-helper factory for Module Node Picker text/status rendering hooks.
 *
 * Purpose:
 *   Centralizes repetitive UI callbacks (process line/action, help text,
 *   custom refresh card tone) and keeps picker composition layer smaller.
 */

import {
    renderHelpText,
    renderHelpHintText,
    renderHelpHintTextWithTone,
    renderHelpModuleSummary,
    renderHelpModuleCardHint,
} from "../../ui/module_node_picker_help.js";

/**
 * Create view helper callbacks for picker orchestration flows.
 */
export function createModuleNodePickerViewHelpers(context = {}) {
    const shouldContinue = typeof context?.shouldContinue === "function"
        ? context.shouldContinue
        : () => true;
    const processUi = context?.processUi || null;
    const getActionBusy = typeof context?.getActionBusy === "function"
        ? context.getActionBusy
        : () => false;
    const customAlert = context?.customAlert || null;
    const customAlertText = context?.customAlertText || null;
    const selectionHelp = context?.selectionHelp || null;
    const moduleHelp = context?.moduleHelp || null;
    const marks = context?.marks || {
        updatedMark: "✅",
        remoteUpdateMark: "🟥",
    };

    const setProcessAction = (label, btnText, onClick) => {
        if (!shouldContinue()) {
            return;
        }
        processUi?.setAction?.(label, btnText, onClick, Boolean(getActionBusy()));
    };

    const setRefreshLine = (text, tone = "neutral") => {
        if (!shouldContinue()) {
            return;
        }
        processUi?.setLine?.(text, tone);
    };

    const setCustomRefreshCardLine = (text, tone = "neutral") => {
        if (!shouldContinue()) {
            return;
        }
        if (!customAlert || !customAlertText) {
            return;
        }
        customAlert.style.display = "block";
        customAlert.classList.remove(
            "alexz-mod-picker-status-card--warn",
            "alexz-mod-picker-status-card--ok",
            "alexz-mod-picker-status-card--neutral"
        );
        if (tone === "warn") {
            customAlert.classList.add("alexz-mod-picker-status-card--warn");
        } else if (tone === "ok") {
            customAlert.classList.add("alexz-mod-picker-status-card--ok");
        } else {
            customAlert.classList.add("alexz-mod-picker-status-card--neutral");
        }
        customAlertText.textContent = String(text || "");
    };

    const setHelpText = (text) => {
        if (!shouldContinue()) {
            return;
        }
        renderHelpText(selectionHelp, text);
    };

    const setHelpHintText = (text, tone = "neutral") => {
        if (!shouldContinue()) {
            return;
        }
        if (String(tone || "").toLowerCase() === "warn") {
            renderHelpHintTextWithTone(selectionHelp, text, "warn");
            return;
        }
        renderHelpHintText(selectionHelp, text);
    };

    const setHelpModuleSummary = (moduleName, nodeCount) => {
        if (!shouldContinue()) {
            return;
        }
        renderHelpModuleSummary(moduleHelp, moduleName, nodeCount, marks);
    };

    const setHelpModuleCardHint = (moduleName, nodeCount) => {
        if (!shouldContinue()) {
            return;
        }
        renderHelpModuleCardHint(moduleHelp, moduleName, nodeCount);
    };

    return {
        setProcessAction,
        setRefreshLine,
        setCustomRefreshCardLine,
        setHelpText,
        setHelpHintText,
        setHelpModuleSummary,
        setHelpModuleCardHint,
    };
}
