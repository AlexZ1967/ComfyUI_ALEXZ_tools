/**
 * Module: web/ui/module_node_picker_help.js
 * Author: AlexZ1967
 * Last updated: 2026-02-10
 *
 * Description:
 *   Help panel rendering helpers for Module Node Picker.
 *
 * Purpose:
 *   Keeps help-content rendering isolated from picker control flow.
 */

/**
 * Replace help area with plain status/help text.
 */
export function renderHelpText(helpEl, text) {
    if (!helpEl) {
        return;
    }
    helpEl.innerHTML = "";
    helpEl.textContent = text || "";
}

/**
 * Render expanded-module help summary with insertion hints and legend.
 */
export function renderHelpModuleSummary(helpEl, moduleName, nodeCount, marks = {}) {
    if (!helpEl) {
        return;
    }
    helpEl.innerHTML = "";

    const hint1 = document.createElement("div");
    hint1.className = "alexz-mod-picker-help-hint";
    hint1.textContent = "Кликните карточку модуля, чтобы скрыть список нод.";
    helpEl.appendChild(hint1);
}

/**
 * Render collapsed-module hint shown before node list expansion.
 */
export function renderHelpModuleCardHint(helpEl, moduleName, nodeCount) {
    if (!helpEl) {
        return;
    }
    helpEl.innerHTML = "";

    const hint = document.createElement("div");
    hint.className = "alexz-mod-picker-help-hint";
    hint.textContent = "Кликните карточку модуля, чтобы показать список нод.";
    helpEl.appendChild(hint);
}
