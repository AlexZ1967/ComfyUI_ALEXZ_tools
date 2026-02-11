/**
 * Module: web/ui/module_node_picker_renderers.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   DOM render helpers for Module Node Picker module and node cards.
 *
 * Purpose:
 *   Keeps large UI rendering blocks isolated from picker orchestration logic.
 */

/**
 * Render node cards for the selected module and bind node insertion actions.
 */
export function renderNodeListPanel(context) {
    const nodeListEl = context?.nodeListEl;
    const selectedModule = String(context?.selectedModule || "");
    const getNodesForSelectedGroup = context?.getNodesForSelectedGroup;
    const expandedModule = String(context?.expandedModule || "");
    const setHelpText = context?.setHelpText;
    const setHelpHintText = context?.setHelpHintText;
    const setHelpModuleCardHint = context?.setHelpModuleCardHint;
    const setHelpModuleSummary = context?.setHelpModuleSummary;
    const moduleNodeDiffs = context?.moduleNodeDiffs;
    const createNodeByInfo = context?.createNodeByInfo;
    const app = context?.app;
    const centerNode = context?.centerNode;
    const marks = context?.marks || {};

    if (!nodeListEl) {
        return;
    }
    nodeListEl.innerHTML = "";

    const allNodes = typeof getNodesForSelectedGroup === "function"
        ? (getNodesForSelectedGroup() || [])
        : [];
    const nodes = allNodes.filter((node) => (node.module || "unknown") === selectedModule);

    if (selectedModule === "-1") {
        setHelpText?.("Выберите модуль, чтобы увидеть список нод.");
        return;
    }
    if (!nodes.length) {
        setHelpHintText?.("Загруженных нод не найдено (возможно, модуль не загрузился).", "warn");
        return;
    }
    if (expandedModule !== selectedModule) {
        setHelpModuleCardHint?.(selectedModule, nodes.length);
        return;
    }

    setHelpModuleSummary?.(selectedModule, nodes.length);
    const nodeDiff = moduleNodeDiffs?.get?.(selectedModule) || {
        newNodes: new Set(),
        updatedNodes: new Set(),
        markAllUpdated: false,
    };

    const legend = document.createElement("div");
    legend.className = "alexz-mod-picker-node-legend";

    const nodeColorRow = document.createElement("div");
    nodeColorRow.className = "alexz-mod-picker-node-legend-row";
    nodeColorRow.append("Рамка ноды: ");
    const redWord = document.createElement("span");
    redWord.className = "alexz-mod-picker-legend-color-red";
    redWord.textContent = "красная";
    nodeColorRow.appendChild(redWord);
    nodeColorRow.append(" = новая, ");
    const greenWord = document.createElement("span");
    greenWord.className = "alexz-mod-picker-legend-color-green";
    greenWord.textContent = "зеленая";
    nodeColorRow.appendChild(greenWord);
    nodeColorRow.append(" = обновленная, ");
    const yellowWord = document.createElement("span");
    yellowWord.className = "alexz-mod-picker-legend-color-yellow";
    yellowWord.textContent = "желтая";
    nodeColorRow.appendChild(yellowWord);
    nodeColorRow.append(" = не определенная.");
    legend.appendChild(nodeColorRow);
    nodeListEl.appendChild(legend);

    const groupEl = document.createElement("div");
    groupEl.className = "alexz-mod-picker-group";

    const groupTitle = document.createElement("div");
    groupTitle.className = "alexz-mod-picker-group-title";
    groupTitle.textContent = `${selectedModule} (${nodes.length})`;
    groupEl.appendChild(groupTitle);

    for (const nodeInfo of nodes) {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "alexz-mod-picker-node";
        if (nodeDiff.markAllUpdated) {
            item.classList.add("alexz-mod-picker-node--updated");
        } else if (nodeDiff.newNodes.has(nodeInfo.node_name)) {
            item.classList.add("alexz-mod-picker-node--new");
        } else if (nodeDiff.updatedNodes.has(nodeInfo.node_name)) {
            item.classList.add("alexz-mod-picker-node--updated");
        }
        item.onclick = () => {
            if (typeof createNodeByInfo !== "function") {
                return;
            }
            const node = createNodeByInfo(nodeInfo);
            if (!node) {
                setHelpText?.(`Не удалось создать ноду: ${nodeInfo.display_name}`);
                return;
            }
            app?.graph?.add(node);
            centerNode?.(node);
            app?.canvas?.selectNode?.(node, false);
            app?.graph?.setDirtyCanvas(true, true);
        };

        const nameEl = document.createElement("div");
        nameEl.className = "alexz-mod-picker-node-name";
        nameEl.textContent = nodeInfo.display_name;
        item.appendChild(nameEl);

        const descEl = document.createElement("div");
        descEl.className = "alexz-mod-picker-node-desc";
        descEl.textContent = `${nodeInfo.annotation} [${nodeInfo.category || "unknown"}]`;
        item.appendChild(descEl);

        groupEl.appendChild(item);
    }
    nodeListEl.appendChild(groupEl);
}

/**
 * Render selected-module metadata card with actions and novelty markers.
 */
export function renderModuleInfoCard(context) {
    const moduleInfoEl = context?.moduleInfoEl;
    const info = context?.info || null;
    const selectedModule = String(context?.selectedModule || "");
    const nodeCount = Number(context?.nodeCount || 0);
    const isModuleUpdated = Boolean(context?.isModuleUpdated);
    const actionBusy = Boolean(context?.actionBusy);
    const inlineStatus = context?.inlineStatus || null;
    const fmtDate = context?.fmtDate;
    const onExpandModule = context?.onExpandModule;
    const onRefreshModuleInfo = context?.onRefreshModuleInfo;

    if (!moduleInfoEl) {
        return;
    }
    moduleInfoEl.innerHTML = "";
    if (!info || selectedModule === "-1") {
        return;
    }

    const card = document.createElement("div");
    card.className = "alexz-mod-picker-module-card";
    const updateStatus = String(info.update_status || "unknown");
    if (isModuleUpdated) {
        card.classList.add("alexz-mod-picker-module-card--updated");
    }
    if (updateStatus === "unknown") {
        card.classList.add("alexz-mod-picker-module-card--unknown");
    }
    if (nodeCount > 0) {
        card.classList.add("alexz-mod-picker-module-card--clickable");
        card.title = "Кликните, чтобы показать или скрыть список нод";
        card.onclick = () => onExpandModule?.(selectedModule);
    }

    const titleEl = document.createElement("div");
    titleEl.className = "alexz-mod-picker-module-title";
    titleEl.textContent = info.title || info.module || selectedModule;
    card.appendChild(titleEl);

    const authorEl = document.createElement("div");
    authorEl.className = "alexz-mod-picker-module-meta";
    const ownerUrl = String(info.owner_url || "").trim();
    if (info.author && ownerUrl) {
        authorEl.append("Owner: ");
        const link = document.createElement("a");
        link.href = ownerUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = String(info.author);
        link.addEventListener("click", (event) => event.stopPropagation());
        authorEl.appendChild(link);
    } else {
        authorEl.textContent = `Owner: ${info.author || "unknown"}`;
    }
    card.appendChild(authorEl);

    if (info.description) {
        const descEl = document.createElement("div");
        descEl.className = "alexz-mod-picker-module-desc";
        descEl.textContent = info.description;
        card.appendChild(descEl);
    }

    const formatDate = typeof fmtDate === "function"
        ? fmtDate
        : (value) => String(value || "n/a");
    const hasInstalledMeta = Boolean(info.installed_updated_at || info.installed_commit_short);
    if (hasInstalledMeta) {
        const installedRow = document.createElement("div");
        installedRow.className = "alexz-mod-picker-module-row";
        const labelEl = document.createElement("span");
        labelEl.className = "alexz-mod-picker-module-label";
        labelEl.textContent = "Installed:";
        const valueEl = document.createElement("span");
        valueEl.textContent = `${info.installed_commit_short ? `${info.installed_commit_short} · ` : ""}${formatDate(info.installed_updated_at)}`;
        installedRow.appendChild(labelEl);
        installedRow.appendChild(valueEl);
        card.appendChild(installedRow);
    }

    if (info.remote_updated_at) {
        const remoteRow = document.createElement("div");
        remoteRow.className = "alexz-mod-picker-module-row";
        const labelEl = document.createElement("span");
        labelEl.className = "alexz-mod-picker-module-label";
        labelEl.textContent = "Remote updated:";
        const valueEl = document.createElement("span");
        valueEl.textContent = formatDate(info.remote_updated_at);
        remoteRow.appendChild(labelEl);
        remoteRow.appendChild(valueEl);
        card.appendChild(remoteRow);
    }

    const isCustomGroup = String(info.group || "") === "custom";
    if (isCustomGroup) {
        const requirementsPending = Boolean(info.requirements_update_pending);
        const requirementsPendingAt = info.requirements_pending_updated_at
            ? ` (${formatDate(info.requirements_pending_updated_at)})`
            : "";

        const statusRow = document.createElement("div");
        statusRow.className = "alexz-mod-picker-module-row";
        const labelEl = document.createElement("span");
        labelEl.className = "alexz-mod-picker-module-label";
        labelEl.textContent = "Status:";
        const valueEl = document.createElement("span");
        if (updateStatus === "can_update") {
            statusRow.classList.add("warn");
            valueEl.textContent = "модуль требует обновления";
        } else if (updateStatus === "up_to_date") {
            statusRow.classList.add("ok");
            valueEl.textContent = "модуль актуален";
        } else {
            statusRow.classList.add("unknown");
            valueEl.textContent = "update status unknown (no upstream/remote)";
        }
        statusRow.appendChild(labelEl);
        statusRow.appendChild(valueEl);
        card.appendChild(statusRow);

        if (requirementsPending) {
            const reqRow = document.createElement("div");
            reqRow.className = "alexz-mod-picker-module-row warn";
            const reqLabel = document.createElement("span");
            reqLabel.className = "alexz-mod-picker-module-label";
            reqLabel.textContent = "Requirements:";
            const reqValue = document.createElement("span");
            reqValue.textContent = `requirements.txt install pending${requirementsPendingAt}`;
            reqRow.appendChild(reqLabel);
            reqRow.appendChild(reqValue);
            card.appendChild(reqRow);
        }
    }

    const actionRow = document.createElement("div");
    actionRow.className = "alexz-mod-picker-action-row";

    const refreshInfoBtn = document.createElement("button");
    refreshInfoBtn.type = "button";
    refreshInfoBtn.className = "alexz-mod-picker-btn-small";
    refreshInfoBtn.textContent = "Обновить информацию о модуле";
    refreshInfoBtn.disabled = actionBusy;
    refreshInfoBtn.onclick = async (event) => {
        event.stopPropagation();
        if (actionBusy || typeof onRefreshModuleInfo !== "function") {
            return;
        }
        const moduleName = String(info.module || selectedModule || "").trim();
        await onRefreshModuleInfo(moduleName, isCustomGroup);
    };
    actionRow.appendChild(refreshInfoBtn);

    card.appendChild(actionRow);

    if (info.new_module_between_runs) {
        const newRow = document.createElement("div");
        newRow.className = "alexz-mod-picker-module-row notice";
        const labelEl = document.createElement("span");
        labelEl.className = "alexz-mod-picker-module-label";
        labelEl.textContent = "Detected between runs:";
        const valueEl = document.createElement("span");
        valueEl.textContent = "new module";
        newRow.appendChild(labelEl);
        newRow.appendChild(valueEl);
        card.appendChild(newRow);
    }

    if (info.updated_between_runs) {
        const updateRow = document.createElement("div");
        updateRow.className = "alexz-mod-picker-module-row notice";
        const labelEl = document.createElement("span");
        labelEl.className = "alexz-mod-picker-module-label";
        labelEl.textContent = "Updated between runs:";
        const valueEl = document.createElement("span");
        const prev = info.startup_prev_commit_short || "unknown";
        const next = info.startup_new_commit_short || "unknown";
        const at = info.startup_update_at ? ` (${formatDate(info.startup_update_at)})` : "";
        if (info.startup_prev_commit_short || info.startup_new_commit_short) {
            valueEl.textContent = `${prev} -> ${next}${at}`;
        } else {
            valueEl.textContent = `local changes detected${at}`;
        }
        updateRow.appendChild(labelEl);
        updateRow.appendChild(valueEl);
        card.appendChild(updateRow);
    }

    const updatedNodes = Array.isArray(info.updated_nodes_between_runs)
        ? info.updated_nodes_between_runs.filter(Boolean)
        : [];
    if (updatedNodes.length) {
        const updatedLine = document.createElement("div");
        updatedLine.className = "alexz-mod-picker-module-note";
        updatedLine.textContent = `Обновлены ноды: ${updatedNodes.join(", ")}`;
        card.appendChild(updatedLine);
    }

    const newNodes = Array.isArray(info.new_nodes_between_runs)
        ? info.new_nodes_between_runs.filter(Boolean)
        : [];
    if (newNodes.length) {
        const newLine = document.createElement("div");
        newLine.className = "alexz-mod-picker-module-note";
        newLine.textContent = `Добавлены ноды: ${newNodes.join(", ")}`;
        card.appendChild(newLine);
    }

    if (inlineStatus?.text) {
        const statusLine = document.createElement("div");
        statusLine.className = "alexz-mod-picker-module-note";
        if (inlineStatus.tone === "ok") {
            statusLine.classList.add("alexz-mod-picker-module-note--ok");
        } else if (inlineStatus.tone === "warn") {
            statusLine.classList.add("alexz-mod-picker-module-note--warn");
        }
        statusLine.textContent = inlineStatus.text;
        card.appendChild(statusLine);
    }

    moduleInfoEl.appendChild(card);
}
