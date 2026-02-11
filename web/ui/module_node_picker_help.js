/**
 * Module: web/ui/module_node_picker_help.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
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
    const main = document.createElement("div");
    main.className = "alexz-mod-picker-help-main";
    main.textContent = text || "";
    helpEl.appendChild(main);
}

/**
 * Replace help area with compact hint-style text.
 */
export function renderHelpHintText(helpEl, text) {
    return renderHelpHintTextWithTone(helpEl, text, "neutral");
}

/**
 * Replace help area with compact hint-style text and tone.
 */
export function renderHelpHintTextWithTone(helpEl, text, tone = "neutral") {
    if (!helpEl) {
        return;
    }
    helpEl.innerHTML = "";
    const hint = document.createElement("div");
    hint.className = "alexz-mod-picker-help-hint";
    if (String(tone || "").toLowerCase() === "warn") {
        hint.classList.add("alexz-mod-picker-help-hint--warn");
    }
    hint.textContent = text || "";
    helpEl.appendChild(hint);
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
