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
    const updatedMark = String(marks.updatedMark || "✅");
    const remoteUpdateMark = String(marks.remoteUpdateMark || "🟥");

    helpEl.innerHTML = "";

    const main = document.createElement("div");
    main.className = "alexz-mod-picker-help-main";
    main.append("Модуль ");
    const moduleStrong = document.createElement("strong");
    moduleStrong.textContent = String(moduleName || "unknown");
    main.appendChild(moduleStrong);
    main.append(": нод ");
    const countStrong = document.createElement("strong");
    countStrong.textContent = String(Math.max(0, Number(nodeCount) || 0));
    main.appendChild(countStrong);
    main.append(".");
    helpEl.appendChild(main);

    const hint1 = document.createElement("div");
    hint1.className = "alexz-mod-picker-help-hint";
    hint1.textContent = "Кликните ноду для вставки в граф.";
    helpEl.appendChild(hint1);

    const hint2 = document.createElement("div");
    hint2.className = "alexz-mod-picker-help-hint";
    hint2.textContent = `Метки модулей: ${updatedMark} обновлен между запусками, ${remoteUpdateMark} доступно обновление.`;
    helpEl.appendChild(hint2);

    const hint3 = document.createElement("div");
    hint3.className = "alexz-mod-picker-help-hint";
    hint3.textContent = "Рамка ноды: красная = новая, зеленая = обновленная.";
    helpEl.appendChild(hint3);
}

/**
 * Render collapsed-module hint shown before node list expansion.
 */
export function renderHelpModuleCardHint(helpEl, moduleName, nodeCount) {
    if (!helpEl) {
        return;
    }
    helpEl.innerHTML = "";

    const main = document.createElement("div");
    main.className = "alexz-mod-picker-help-main";
    main.append("Модуль ");
    const moduleStrong = document.createElement("strong");
    moduleStrong.textContent = String(moduleName || "unknown");
    main.appendChild(moduleStrong);
    main.append(": нод ");
    const countStrong = document.createElement("strong");
    countStrong.textContent = String(Math.max(0, Number(nodeCount) || 0));
    main.appendChild(countStrong);
    main.append(".");
    helpEl.appendChild(main);

    const hint = document.createElement("div");
    hint.className = "alexz-mod-picker-help-hint";
    hint.textContent = "Кликните карточку модуля, чтобы показать список нод.";
    helpEl.appendChild(hint);
}
