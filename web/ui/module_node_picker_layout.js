/**
 * Module: web/ui/module_node_picker_layout.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   DOM layout factory for Module Node Picker panel.
 *
 * Purpose:
 *   Creates and wires static panel elements in one place, returning a stable
 *   set of element references for picker logic/orchestration layers.
 */

/**
 * Build Module Node Picker layout and return key element references.
 */
export function createModuleNodePickerLayout(container) {
    container.innerHTML = "";

    const root = document.createElement("div");
    root.className = "alexz-mod-picker";
    container.appendChild(root);

    const head = document.createElement("div");
    head.className = "alexz-mod-picker-head";
    root.appendChild(head);

    const title = document.createElement("div");
    title.className = "alexz-mod-picker-title";
    title.textContent = "Node Picker";
    head.appendChild(title);

    const warmupHint = document.createElement("span");
    warmupHint.className = "alexz-mod-picker-title-warmup";
    warmupHint.textContent = "warming up...";
    warmupHint.style.display = "none";
    head.appendChild(warmupHint);

    const headRight = document.createElement("div");
    headRight.className = "alexz-mod-picker-head-right";
    head.appendChild(headRight);

    const debugToggle = document.createElement("button");
    debugToggle.type = "button";
    debugToggle.className = "alexz-mod-picker-btn-small";
    debugToggle.textContent = "Debug";
    headRight.appendChild(debugToggle);

    const debugCard = document.createElement("div");
    debugCard.className = "alexz-mod-picker-debug-card";
    root.appendChild(debugCard);

    const debugCardHeader = document.createElement("div");
    debugCardHeader.className = "alexz-mod-picker-debug-card-header";
    debugCard.appendChild(debugCardHeader);

    const debugTitle = document.createElement("div");
    debugTitle.className = "alexz-mod-picker-debug-title";
    debugTitle.textContent = "Debug diagnostics";
    debugCardHeader.appendChild(debugTitle);

    const debugCopyBtn = document.createElement("button");
    debugCopyBtn.type = "button";
    debugCopyBtn.className = "alexz-mod-picker-btn-small";
    debugCopyBtn.textContent = "Copy ⧉";
    debugCardHeader.appendChild(debugCopyBtn);

    const diagnostics = document.createElement("div");
    diagnostics.className = "alexz-mod-picker-diag";
    diagnostics.textContent = "diag: waiting for sidebar sync...";
    debugCard.appendChild(diagnostics);

    const dividerTop = document.createElement("div");
    dividerTop.className = "alexz-mod-picker-divider";
    root.appendChild(dividerTop);

    const updateGrid = document.createElement("div");
    updateGrid.className = "alexz-mod-picker-update-grid";
    root.appendChild(updateGrid);

    const comfyUpdateBlock = document.createElement("div");
    comfyUpdateBlock.className = "alexz-mod-picker-update-block";
    updateGrid.appendChild(comfyUpdateBlock);

    const customUpdateBlock = document.createElement("div");
    customUpdateBlock.className = "alexz-mod-picker-update-block";
    updateGrid.appendChild(customUpdateBlock);

    const comfyInfoBtn = document.createElement("button");
    comfyInfoBtn.type = "button";
    comfyInfoBtn.textContent = "Refresh ComfyUI Info";
    comfyInfoBtn.className = "alexz-mod-picker-btn-small";
    comfyUpdateBlock.appendChild(comfyInfoBtn);

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.textContent = "Refresh Custom Nodes Info";
    refreshBtn.className = "alexz-mod-picker-btn-small";
    customUpdateBlock.appendChild(refreshBtn);

    const comfyModeSelect = document.createElement("select");
    comfyModeSelect.className = "alexz-mod-picker-btn-small";
    comfyModeSelect.title = "ComfyUI update-check mode";
    const modeReleases = document.createElement("option");
    modeReleases.value = "releases";
    modeReleases.textContent = "ComfyUI check: releases";
    comfyModeSelect.appendChild(modeReleases);
    const modeCommits = document.createElement("option");
    modeCommits.value = "commits";
    modeCommits.textContent = "ComfyUI check: commits";
    comfyModeSelect.appendChild(modeCommits);
    comfyUpdateBlock.appendChild(comfyModeSelect);

    const comfyAlert = document.createElement("div");
    comfyAlert.className = "alexz-mod-picker-status-card alexz-mod-picker-status-card--neutral";
    comfyAlert.style.display = "none";
    const comfyAlertText = document.createElement("div");
    comfyAlert.appendChild(comfyAlertText);
    root.appendChild(comfyAlert);

    const customAlert = document.createElement("div");
    customAlert.className = "alexz-mod-picker-status-card alexz-mod-picker-status-card--neutral";
    customAlert.style.display = "none";
    const customAlertText = document.createElement("div");
    customAlert.appendChild(customAlertText);
    root.appendChild(customAlert);

    const processHost = document.createElement("div");
    processHost.className = "alexz-mod-picker-process-inline";

    const dividerBottom = document.createElement("div");
    dividerBottom.className = "alexz-mod-picker-divider";
    root.appendChild(dividerBottom);

    const selectionLegendHint = document.createElement("div");
    selectionLegendHint.className = "alexz-mod-picker-help-hint alexz-mod-picker-help-hint--selection-legend";
    selectionLegendHint.textContent = "Маркеры состояния:\n✅ локально обновлен, 🟥 доступно обновление, 🟨 статус не определен";
    root.appendChild(selectionLegendHint);

    const selectionHelp = document.createElement("div");
    selectionHelp.className = "alexz-mod-picker-help alexz-mod-picker-help--selection";
    root.appendChild(selectionHelp);

    const categorySelect = document.createElement("select");
    categorySelect.className = "alexz-mod-picker-select";
    const catComfy = document.createElement("option");
    catComfy.value = "comfy";
    catComfy.textContent = "ComfyUI Nodes";
    categorySelect.appendChild(catComfy);
    const catCustom = document.createElement("option");
    catCustom.value = "custom";
    catCustom.textContent = "Custom Nodes";
    categorySelect.appendChild(catCustom);
    categorySelect.value = "custom";
    root.appendChild(categorySelect);

    const groupSelect = document.createElement("select");
    groupSelect.className = "alexz-mod-picker-select";
    root.appendChild(groupSelect);

    const nodeSelect = document.createElement("select");
    nodeSelect.className = "alexz-mod-picker-select";
    root.appendChild(nodeSelect);

    const moduleFilter = document.createElement("input");
    moduleFilter.type = "text";
    moduleFilter.className = "alexz-mod-picker-select";
    moduleFilter.placeholder = "Фильтр модулей (например: Inpaint-Crop)";
    root.appendChild(moduleFilter);

    const moduleHintDivider = document.createElement("div");
    moduleHintDivider.className = "alexz-mod-picker-divider";
    root.appendChild(moduleHintDivider);

    const refreshLine = document.createElement("div");
    refreshLine.className = "alexz-mod-picker-refresh-line";
    processHost.appendChild(refreshLine);

    const processActions = document.createElement("div");
    processActions.className = "alexz-mod-picker-status-card-actions";
    processHost.appendChild(processActions);

    const moduleInfoWrap = document.createElement("div");
    moduleInfoWrap.className = "alexz-mod-picker-module-info-wrap";
    root.appendChild(moduleInfoWrap);

    const moduleHelp = document.createElement("div");
    moduleHelp.className = "alexz-mod-picker-help alexz-mod-picker-help--module";
    moduleInfoWrap.appendChild(moduleHelp);

    const moduleInfo = document.createElement("div");
    moduleInfoWrap.appendChild(moduleInfo);

    const nodeList = document.createElement("div");
    root.appendChild(nodeList);

    return {
        root,
        head,
        title,
        warmupHint,
        headRight,
        debugToggle,
        debugCard,
        debugCardHeader,
        debugTitle,
        debugCopyBtn,
        diagnostics,
        dividerTop,
        updateGrid,
        comfyUpdateBlock,
        customUpdateBlock,
        comfyInfoBtn,
        refreshBtn,
        comfyModeSelect,
        comfyAlert,
        comfyAlertText,
        customAlert,
        customAlertText,
        processHost,
        dividerBottom,
        selectionLegendHint,
        selectionHelp,
        categorySelect,
        groupSelect,
        nodeSelect,
        moduleFilter,
        moduleHintDivider,
        refreshLine,
        processActions,
        moduleInfoWrap,
        moduleHelp,
        moduleInfo,
        nodeList,
    };
}
