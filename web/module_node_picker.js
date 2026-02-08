import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const SIDEBAR_TAB_ID = "alexz-module-nodes";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const GROUP_LABELS = {
    core: "Core_Nodes",
    core_extras: "Core_Extras_Nodes",
    api: "API_Nodes",
    custom: "Custom_Nodes",
};

function injectStyles() {
    const styleId = "alexz-module-picker-style";
    if (document.getElementById(styleId)) {
        return;
    }
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
    .alexz-mod-picker {
        padding: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        height: 100%;
        overflow: auto;
    }
    .alexz-mod-picker-head {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .alexz-mod-picker-title {
        font-size: 13px;
        font-weight: 700;
        opacity: 0.95;
        margin-right: auto;
    }
    .alexz-mod-picker-select {
        width: 100%;
    }
    .alexz-mod-picker-help {
        font-size: 12px;
        opacity: 0.8;
    }
    .alexz-mod-picker-module-card {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .alexz-mod-picker-module-title {
        font-size: 12px;
        font-weight: 700;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta {
        font-size: 11px;
        opacity: 0.9;
        word-break: break-all;
    }
    .alexz-mod-picker-module-meta a {
        color: var(--link-color, #87b5ff);
        text-decoration: underline;
    }
    .alexz-mod-picker-module-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        white-space: pre-wrap;
    }
    .alexz-mod-picker-module-row {
        font-size: 11px;
        opacity: 0.9;
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
    }
    .alexz-mod-picker-module-row.notice {
        color: #f0b429;
    }
    .alexz-mod-picker-module-label {
        font-weight: 700;
        opacity: 0.95;
    }
    .alexz-mod-picker-status {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border-color, #555);
        border-radius: 10px;
        padding: 1px 7px;
        font-size: 10px;
        line-height: 1.4;
        font-weight: 700;
    }
    .alexz-mod-picker-status.up-to-date {
        color: #3dbb7e;
    }
    .alexz-mod-picker-status.can-update {
        color: #f0b429;
    }
    .alexz-mod-picker-status.unknown {
        color: #b3b3b3;
    }
    .alexz-mod-picker-group {
        border: 1px solid var(--border-color, #444);
        border-radius: 7px;
        padding: 8px;
        margin-top: 2px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
    }
    .alexz-mod-picker-group-title {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 7px;
        word-break: break-all;
    }
    .alexz-mod-picker-node {
        width: 100%;
        text-align: left;
        margin-bottom: 6px;
        border: 1px solid var(--border-color, #555);
        border-radius: 6px;
        background: var(--comfy-input-bg, rgba(255,255,255,0.03));
        padding: 7px;
        cursor: pointer;
    }
    .alexz-mod-picker-node:hover {
        filter: brightness(1.12);
    }
    .alexz-mod-picker-node-name {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 3px;
    }
    .alexz-mod-picker-node-desc {
        font-size: 11px;
        opacity: 0.85;
        line-height: 1.28em;
        word-break: break-all;
    }
    .alexz-mod-picker-floating-btn {
        position: fixed;
        left: 10px;
        bottom: 10px;
        z-index: 10005;
    }`;
    document.head.appendChild(style);
}

function centerNode(node) {
    const area = app.canvas?.visible_area;
    if (area && area.length >= 4) {
        node.pos = [
            area[0] + area[2] * 0.5 - node.size[0] * 0.5,
            area[1] + area[3] * 0.5 - node.size[1] * 0.5,
        ];
    } else {
        node.pos = [200, 120];
    }
}

function createNodeByInfo(nodeInfo) {
    const candidates = [nodeInfo.node_name, nodeInfo.display_name].filter(Boolean);
    for (const name of candidates) {
        const node = LiteGraph.createNode(name);
        if (node) {
            return node;
        }
    }
    return null;
}

async function fetchNodeCatalog() {
    const resp = await api.fetchApi("/alexz_tools/node_catalog", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

async function fetchModuleInfo(group, moduleName) {
    const resp = await api.fetchApi(
        `/alexz_tools/module_info?group=${encodeURIComponent(group || "")}&module=${encodeURIComponent(moduleName || "")}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

function fmtDate(iso) {
    if (!iso) {
        return "n/a";
    }
    try {
        return new Date(iso).toLocaleString();
    } catch (err) {
        return String(iso);
    }
}

function statusUi(info) {
    const status = String(info?.update_status || "unknown");
    if (status === "can_update") {
        return { label: "Update available", cls: "can-update" };
    }
    if (status === "up_to_date") {
        return { label: "Up to date", cls: "up-to-date" };
    }
    return { label: "Unknown", cls: "unknown" };
}

function renderPicker(container) {
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

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.textContent = "Обновить";
    head.appendChild(refreshBtn);

    const groupSelect = document.createElement("select");
    groupSelect.className = "alexz-mod-picker-select";
    root.appendChild(groupSelect);

    const nodeSelect = document.createElement("select");
    nodeSelect.className = "alexz-mod-picker-select";
    root.appendChild(nodeSelect);

    const help = document.createElement("div");
    help.className = "alexz-mod-picker-help";
    root.appendChild(help);

    const moduleInfo = document.createElement("div");
    root.appendChild(moduleInfo);

    const nodeList = document.createElement("div");
    root.appendChild(nodeList);

    const catalogByGroup = new Map();

    const getNodesForSelectedGroup = () => {
        const group = groupSelect.value;
        return catalogByGroup.get(group) || [];
    };

    const fillModuleSelect = () => {
        const nodes = getNodesForSelectedGroup();
        nodeSelect.innerHTML = "";
        const grouped = new Map();
        for (const node of nodes) {
            const moduleName = node.module || "unknown";
            if (!grouped.has(moduleName)) {
                grouped.set(moduleName, []);
            }
            grouped.get(moduleName).push(node);
        }
        const modules = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
        if (modules.length === 0) {
            const empty = document.createElement("option");
            empty.value = "-1";
            empty.textContent = "В этой группе нет модулей";
            nodeSelect.appendChild(empty);
            nodeSelect.value = "-1";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
            help.textContent = "Модули не найдены для выбранной группы.";
            return;
        }
        for (const moduleName of modules) {
            const opt = document.createElement("option");
            opt.value = moduleName;
            opt.textContent = `${moduleName} (${(grouped.get(moduleName) || []).length})`;
            nodeSelect.appendChild(opt);
        }
        if (modules.includes(DEFAULT_MODULE)) {
            nodeSelect.value = DEFAULT_MODULE;
        } else {
            nodeSelect.value = modules[0];
        }
        renderNodeList();
        loadModuleInfo();
    };

    const fillGroupSelect = (groups) => {
        groupSelect.innerHTML = "";
        groups.forEach((group) => {
            const opt = document.createElement("option");
            opt.value = group.id;
            const label = GROUP_LABELS[group.id] || group.title || group.id;
            opt.textContent = `${label} (${group.count})`;
            groupSelect.appendChild(opt);
            catalogByGroup.set(group.id, group.nodes || []);
        });

        if (catalogByGroup.has("custom")) {
            groupSelect.value = "custom";
        } else if (groups.length > 0) {
            groupSelect.value = groups[0].id;
        }
        fillModuleSelect();
    };

    const renderNodeList = () => {
        nodeList.innerHTML = "";
        const selectedModule = nodeSelect.value;
        const nodes = getNodesForSelectedGroup().filter(
            (node) => (node.module || "unknown") === selectedModule
        );
        if (!nodes.length || selectedModule === "-1") {
            help.textContent = "Выберите модуль, чтобы увидеть список нод.";
            return;
        }

        help.textContent = `Модуль ${selectedModule}: нод ${nodes.length}. Кликните ноду для вставки в граф.`;

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
            item.onclick = () => {
                const node = createNodeByInfo(nodeInfo);
                if (!node) {
                    help.textContent = `Не удалось создать ноду: ${nodeInfo.display_name}`;
                    return;
                }
                app.graph.add(node);
                centerNode(node);
                app.canvas?.selectNode?.(node, false);
                app.graph.setDirtyCanvas(true, true);
                help.textContent = `Добавлена: ${nodeInfo.display_name}`;
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
        nodeList.appendChild(groupEl);
    };

    const renderModuleInfo = (info) => {
        moduleInfo.innerHTML = "";
        if (!info || nodeSelect.value === "-1") {
            return;
        }

        const card = document.createElement("div");
        card.className = "alexz-mod-picker-module-card";

        const titleEl = document.createElement("div");
        titleEl.className = "alexz-mod-picker-module-title";
        titleEl.textContent = info.title || info.module || nodeSelect.value;
        card.appendChild(titleEl);

        const authorEl = document.createElement("div");
        authorEl.className = "alexz-mod-picker-module-meta";
        if (info.author && info.owner_url) {
            authorEl.innerHTML = `Owner: <a href="${info.owner_url}" target="_blank" rel="noopener noreferrer">${info.author}</a>`;
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

        const hasInstalledMeta = Boolean(info.installed_updated_at || info.installed_commit_short);
        if (hasInstalledMeta) {
            const installedRow = document.createElement("div");
            installedRow.className = "alexz-mod-picker-module-row";
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Installed:";
            const valueEl = document.createElement("span");
            valueEl.textContent = `${info.installed_commit_short ? `${info.installed_commit_short} · ` : ""}${fmtDate(info.installed_updated_at)}`;
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
            valueEl.textContent = fmtDate(info.remote_updated_at);
            remoteRow.appendChild(labelEl);
            remoteRow.appendChild(valueEl);
            card.appendChild(remoteRow);
        }

        if (String(info.group || "") === "custom") {
            const statusRow = document.createElement("div");
            statusRow.className = "alexz-mod-picker-module-row";
            const s = statusUi(info);
            const labelEl = document.createElement("span");
            labelEl.className = "alexz-mod-picker-module-label";
            labelEl.textContent = "Status:";
            const valueEl = document.createElement("span");
            valueEl.className = `alexz-mod-picker-status ${s.cls}`;
            valueEl.textContent = s.label;
            statusRow.appendChild(labelEl);
            statusRow.appendChild(valueEl);
            card.appendChild(statusRow);
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
            const at = info.startup_update_at ? ` (${fmtDate(info.startup_update_at)})` : "";
            valueEl.textContent = `${prev} -> ${next}${at}`;
            updateRow.appendChild(labelEl);
            updateRow.appendChild(valueEl);
            card.appendChild(updateRow);
        }

        moduleInfo.appendChild(card);
    };

    const loadModuleInfo = async () => {
        const selectedModule = nodeSelect.value;
        const selectedGroup = groupSelect.value;
        if (!selectedModule || selectedModule === "-1") {
            moduleInfo.innerHTML = "";
            return;
        }
        try {
            const payload = await fetchModuleInfo(selectedGroup, selectedModule);
            if (nodeSelect.value !== selectedModule || groupSelect.value !== selectedGroup) {
                return;
            }
            renderModuleInfo(payload?.info || null);
        } catch (err) {
            moduleInfo.innerHTML = "";
        }
    };

    const loadCatalog = async () => {
        help.textContent = "Загрузка списка нод...";
        try {
            const payload = await fetchNodeCatalog();
            catalogByGroup.clear();
            const groups = payload?.groups || [];
            fillGroupSelect(groups);
            const summary = groups
                .map((group) => {
                    const label = GROUP_LABELS[group.id] || group.title || group.id;
                    return `${label}=${group.count}`;
                })
                .join(", ");
            help.textContent = `Группы: ${summary}.`;
        } catch (err) {
            help.textContent = `Ошибка загрузки: ${String(err)}`;
            groupSelect.innerHTML = "";
            nodeSelect.innerHTML = "";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
        }
    };

    groupSelect.onchange = () => fillModuleSelect();
    nodeSelect.onchange = () => {
        renderNodeList();
        loadModuleInfo();
    };
    refreshBtn.onclick = () => loadCatalog();

    loadCatalog();
}

function activateSidebarTab() {
    const sidebar = app.extensionManager?.sidebarTab || app.extensionManager;
    if (!sidebar) {
        return false;
    }
    if ("activeSidebarTabId" in sidebar) {
        sidebar.activeSidebarTabId = SIDEBAR_TAB_ID;
        return true;
    }
    if ("activeSidebarTab" in sidebar) {
        sidebar.activeSidebarTab = SIDEBAR_TAB_ID;
        return true;
    }
    return false;
}

function attachFallbackButton() {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Module Nodes";
    button.title = "Открыть подбор нод";
    button.onclick = () => {
        if (!activateSidebarTab()) {
            button.textContent = "Sidebar API недоступен";
        }
    };

    const menuContainer = app.ui?.menuContainer;
    if (menuContainer) {
        button.style.width = "100%";
        button.style.order = 95;
        menuContainer.append(button);
        return;
    }

    button.className = "alexz-mod-picker-floating-btn";
    document.body.appendChild(button);
}

app.registerExtension({
    name: EXT_NAME,
    setup() {
        injectStyles();

        if (app.extensionManager && typeof app.extensionManager.registerSidebarTab === "function") {
            app.extensionManager.registerSidebarTab({
                id: SIDEBAR_TAB_ID,
                icon: "pi pi-sitemap",
                title: "Module Nodes",
                tooltip: "Выбор и вставка нод по группам Core/Custom",
                type: "custom",
                render: (container) => {
                    renderPicker(container);
                },
            });
            return;
        }

        attachFallbackButton();
    },
});
