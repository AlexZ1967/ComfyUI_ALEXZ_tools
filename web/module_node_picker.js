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
const MODULE_MARK_UPDATED = "✅";
const MODULE_MARK_REMOTE_UPDATE = "🟥";

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
        display: flex;
        flex-direction: column;
        gap: 3px;
        min-height: 2.2em;
    }
    .alexz-mod-picker-help-main {
        font-size: 13px;
        line-height: 1.3;
        opacity: 0.95;
    }
    .alexz-mod-picker-help-main strong {
        font-weight: 700;
    }
    .alexz-mod-picker-help-hint {
        font-size: 11px;
        line-height: 1.3;
        opacity: 0.78;
        font-style: italic;
    }
    .alexz-mod-picker-refresh-line {
        font-size: 12px;
        opacity: 0.92;
        min-height: 1.2em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--ok {
        color: #3dbb7e;
        font-weight: 700;
    }
    .alexz-mod-picker-refresh-line.alexz-mod-picker-refresh-line--warn {
        color: #ff6b6b;
        font-weight: 700;
    }
    .alexz-mod-picker-comfy-alert {
        border: 1px solid #b64040;
        background: rgba(180, 64, 64, 0.16);
        color: #ff6b6b;
        border-radius: 7px;
        padding: 7px 8px;
        font-size: 12px;
        line-height: 1.3;
        font-weight: 700;
        display: none;
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
    .alexz-mod-picker-node.alexz-mod-picker-node--updated {
        border-color: #3dbb7e;
        box-shadow: inset 0 0 0 1px rgba(61, 187, 126, 0.35);
    }
    .alexz-mod-picker-node.alexz-mod-picker-node--new {
        border-color: #d44f4f;
        box-shadow: inset 0 0 0 1px rgba(212, 79, 79, 0.35);
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

async function refreshModuleRuntimeState() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh", {
        method: "POST",
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

async function fetchModuleRefreshStatus() {
    const resp = await api.fetchApi("/alexz_tools/module_refresh_status", {
        cache: "no-store",
    });
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

function moduleBadgesFromInfo(info) {
    const behind = Number(info?.git_behind);
    return {
        updatedBetweenRuns: Boolean(info?.updated_between_runs),
        hasRemoteUpdate: Number.isFinite(behind) && behind > 0,
    };
}

function formatModuleOption(moduleName, count, badges) {
    const marks = [];
    if (badges?.updatedBetweenRuns) {
        marks.push(MODULE_MARK_UPDATED);
    }
    if (badges?.hasRemoteUpdate) {
        marks.push(MODULE_MARK_REMOTE_UPDATE);
    }
    const prefix = marks.length ? `${marks.join(" ")} ` : "";
    return `${prefix}${moduleName} (${count})`;
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
    refreshBtn.textContent = "Обновить информацию о модулях";
    head.appendChild(refreshBtn);

    const comfyAlert = document.createElement("div");
    comfyAlert.className = "alexz-mod-picker-comfy-alert";
    root.appendChild(comfyAlert);

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

    const refreshLine = document.createElement("div");
    refreshLine.className = "alexz-mod-picker-refresh-line";
    root.appendChild(refreshLine);

    const help = document.createElement("div");
    help.className = "alexz-mod-picker-help";
    root.appendChild(help);

    const moduleInfo = document.createElement("div");
    root.appendChild(moduleInfo);

    const nodeList = document.createElement("div");
    root.appendChild(nodeList);

    const catalogByGroup = new Map();
    const moduleCatalogByGroup = new Map();
    const moduleCounts = new Map();
    const moduleOptions = new Map();
    const moduleBadges = new Map();
    const moduleNodeDiffs = new Map();
    let moduleBadgeLoadToken = 0;
    let refreshPollToken = 0;

    const renderComfyAlert = (info) => {
        const behind = Number(info?.behind);
        const status = String(info?.update_status || "unknown");
        if (status !== "can_update" || !Number.isFinite(behind) || behind <= 0) {
            comfyAlert.style.display = "none";
            comfyAlert.textContent = "";
            return;
        }
        const branch = String(info?.branch || "unknown");
        const local = String(info?.installed_commit_short || "unknown");
        const remote = String(info?.remote_commit_short || "unknown");
        comfyAlert.textContent = `Доступна новая версия ComfyUI на GitHub: branch=${branch}, behind=${behind}, local=${local}, remote=${remote}.`;
        comfyAlert.style.display = "block";
    };

    const getNodesForSelectedGroup = () => {
        const group = groupSelect.value;
        return catalogByGroup.get(group) || [];
    };

    const setRefreshLine = (text, tone = "neutral") => {
        refreshLine.textContent = text || "";
        refreshLine.classList.remove("alexz-mod-picker-refresh-line--ok", "alexz-mod-picker-refresh-line--warn");
        if (tone === "ok") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--ok");
        } else if (tone === "warn") {
            refreshLine.classList.add("alexz-mod-picker-refresh-line--warn");
        }
    };

    const setHelpText = (text) => {
        help.innerHTML = "";
        help.textContent = text || "";
    };

    const setHelpModuleSummary = (moduleName, nodeCount) => {
        help.innerHTML = "";

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
        help.appendChild(main);

        const hint1 = document.createElement("div");
        hint1.className = "alexz-mod-picker-help-hint";
        hint1.textContent = "Кликните ноду для вставки в граф.";
        help.appendChild(hint1);

        const hint2 = document.createElement("div");
        hint2.className = "alexz-mod-picker-help-hint";
        hint2.textContent = `Метки модулей: ${MODULE_MARK_UPDATED} обновлен между запусками, ${MODULE_MARK_REMOTE_UPDATE} доступно обновление.`;
        help.appendChild(hint2);

        const hint3 = document.createElement("div");
        hint3.className = "alexz-mod-picker-help-hint";
        hint3.textContent = "Рамка ноды: красная = новая, зеленая = обновленная.";
        help.appendChild(hint3);
    };

    const formatRefreshLine = (refresh) => {
        const phase = String(refresh?.phase || "");
        const current = Number(refresh?.current || 0);
        const total = Number(refresh?.total || 0);
        const remaining = Number(refresh?.remaining || 0);
        const modulesNeedUpdate = Number(refresh?.modules_need_update || 0);
        const moduleName = String(refresh?.module || "");
        const error = String(refresh?.error || "");

        if (phase === "sync") {
            if (total > 0) {
                const modulePart = moduleName ? ` (${moduleName})` : "";
                return { text: `Обновление статусов модулей: ${current}/${total}, осталось ${remaining}${modulePart}`, tone: "neutral" };
            }
            return { text: "Обновление статусов модулей: подготовка...", tone: "neutral" };
        }
        if (phase === "snapshots") {
            return { text: "Обновление статусов модулей: пересчет...", tone: "neutral" };
        }
        if (phase === "done") {
            const count = Number.isFinite(modulesNeedUpdate) ? Math.max(0, modulesNeedUpdate) : 0;
            if (count > 0) {
                return { text: `${count} модулей требуют обновления`, tone: "warn" };
            }
            return { text: "обновления не требуются", tone: "ok" };
        }
        if (phase === "error") {
            return { text: `Обновление статусов модулей: ошибка${error ? ` (${error})` : ""}.`, tone: "warn" };
        }
        return { text: "Обновление статусов модулей: запуск...", tone: "neutral" };
    };

    const pollRefreshProgress = async () => {
        const token = ++refreshPollToken;
        while (token === refreshPollToken) {
            let payload;
            try {
                payload = await fetchModuleRefreshStatus();
            } catch (err) {
                setRefreshLine(`Обновление статусов модулей: ошибка статуса (${String(err)}).`, "warn");
                return false;
            }
            const refresh = payload?.refresh || {};
            const line = formatRefreshLine(refresh);
            setRefreshLine(line.text, line.tone);
            if (!refresh?.running) {
                return refresh?.phase !== "error";
            }
            await new Promise((resolve) => setTimeout(resolve, 400));
        }
        return false;
    };

    const setModuleOptionText = (moduleName) => {
        const option = moduleOptions.get(moduleName);
        if (!option) {
            return;
        }
        const count = moduleCounts.get(moduleName) || 0;
        const badges = moduleBadges.get(moduleName) || null;
        option.textContent = formatModuleOption(moduleName, count, badges);
    };

    const setModuleNodeDiffs = (moduleName, info) => {
        const newNodes = Array.isArray(info?.new_nodes_between_runs) ? info.new_nodes_between_runs : [];
        const updatedNodes = Array.isArray(info?.updated_nodes_between_runs) ? info.updated_nodes_between_runs : [];
        moduleNodeDiffs.set(moduleName, {
            newNodes: new Set(newNodes),
            updatedNodes: new Set(updatedNodes),
            markAllUpdated: Boolean(info?.new_module_between_runs),
        });
    };

    const loadModuleBadges = async (group, modules) => {
        const token = ++moduleBadgeLoadToken;
        if (!modules.length) {
            return;
        }

        const queue = [...modules];
        const workers = Array.from({ length: Math.min(4, queue.length) }, async () => {
            while (queue.length && token === moduleBadgeLoadToken) {
                const moduleName = queue.shift();
                if (!moduleName) {
                    break;
                }
                try {
                    const payload = await fetchModuleInfo(group, moduleName);
                    if (token !== moduleBadgeLoadToken || groupSelect.value !== group) {
                        return;
                    }
                    const badges = moduleBadgesFromInfo(payload?.info || {});
                    if (badges.updatedBetweenRuns || badges.hasRemoteUpdate) {
                        moduleBadges.set(moduleName, badges);
                    } else {
                        moduleBadges.delete(moduleName);
                    }
                    setModuleOptionText(moduleName);
                } catch (err) {
                    // Ignore per-module errors and keep list usable.
                }
            }
        });

        await Promise.all(workers);
    };

    const fillModuleSelect = () => {
        const nodes = getNodesForSelectedGroup();
        const selectedGroup = groupSelect.value;
        const moduleEntries = moduleCatalogByGroup.get(selectedGroup) || [];
        const filterValue = (moduleFilter.value || "").trim().toLowerCase();
        moduleBadgeLoadToken += 1;
        moduleCounts.clear();
        moduleOptions.clear();
        moduleBadges.clear();
        moduleNodeDiffs.clear();
        nodeSelect.innerHTML = "";
        const grouped = new Map();
        for (const node of nodes) {
            const moduleName = node.module || "unknown";
            if (!grouped.has(moduleName)) {
                grouped.set(moduleName, []);
            }
            grouped.get(moduleName).push(node);
        }
        let modules = [];
        if (moduleEntries.length) {
            modules = moduleEntries
                .map((entry) => String(entry?.module || "unknown"))
                .sort((a, b) => a.localeCompare(b));
        } else {
            modules = Array.from(grouped.keys()).sort((a, b) => a.localeCompare(b));
        }
        if (filterValue) {
            modules = modules.filter((name) => name.toLowerCase().includes(filterValue));
        }
        if (modules.length === 0) {
            const empty = document.createElement("option");
            empty.value = "-1";
            empty.textContent = filterValue ? "Нет модулей по фильтру" : "В этой группе нет модулей";
            nodeSelect.appendChild(empty);
            nodeSelect.value = "-1";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
            setHelpText(filterValue
                ? `Нет модулей по фильтру: "${moduleFilter.value}".`
                : "Модули не найдены для выбранной группы.");
            return;
        }
        const countMap = new Map();
        for (const entry of moduleEntries) {
            const moduleName = String(entry?.module || "unknown");
            countMap.set(moduleName, Number(entry?.count) || 0);
        }
        for (const moduleName of modules) {
            const opt = document.createElement("option");
            opt.value = moduleName;
            const count = countMap.has(moduleName)
                ? (countMap.get(moduleName) || 0)
                : (grouped.get(moduleName) || []).length;
            moduleCounts.set(moduleName, count);
            moduleOptions.set(moduleName, opt);
            opt.textContent = formatModuleOption(moduleName, count, null);
            nodeSelect.appendChild(opt);
        }
        if (modules.includes(DEFAULT_MODULE)) {
            nodeSelect.value = DEFAULT_MODULE;
        } else {
            nodeSelect.value = modules[0];
        }
        renderNodeList();
        loadModuleInfo();
        loadModuleBadges(selectedGroup, modules);
    };

    const fillGroupSelect = (groups) => {
        groupSelect.innerHTML = "";
        moduleCatalogByGroup.clear();
        groups.forEach((group) => {
            const opt = document.createElement("option");
            opt.value = group.id;
            const label = GROUP_LABELS[group.id] || group.title || group.id;
            opt.textContent = `${label} (${group.count})`;
            groupSelect.appendChild(opt);
            catalogByGroup.set(group.id, group.nodes || []);
            moduleCatalogByGroup.set(group.id, group.modules || []);
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
        if (selectedModule === "-1") {
            setHelpText("Выберите модуль, чтобы увидеть список нод.");
            return;
        }
        if (!nodes.length) {
            setHelpText(`Модуль ${selectedModule}: загруженных нод не найдено (возможно, модуль не загрузился).`);
            return;
        }

        setHelpModuleSummary(selectedModule, nodes.length);
        const nodeDiff = moduleNodeDiffs.get(selectedModule) || {
            newNodes: new Set(),
            updatedNodes: new Set(),
            markAllUpdated: false,
        };

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
                const node = createNodeByInfo(nodeInfo);
                if (!node) {
                    setHelpText(`Не удалось создать ноду: ${nodeInfo.display_name}`);
                    return;
                }
                app.graph.add(node);
                centerNode(node);
                app.canvas?.selectNode?.(node, false);
                app.graph.setDirtyCanvas(true, true);
                setHelpText(`Добавлена: ${nodeInfo.display_name}`);
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

        if (info.updated_between_runs && (info.startup_prev_commit_short || info.startup_new_commit_short)) {
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
            const info = payload?.info || null;
            renderModuleInfo(info);
            if (info) {
                const badges = moduleBadgesFromInfo(info);
                if (badges.updatedBetweenRuns || badges.hasRemoteUpdate) {
                    moduleBadges.set(selectedModule, badges);
                } else {
                    moduleBadges.delete(selectedModule);
                }
                setModuleNodeDiffs(selectedModule, info);
                setModuleOptionText(selectedModule);
                renderNodeList();
            }
        } catch (err) {
            moduleInfo.innerHTML = "";
        }
    };

    const loadCatalog = async () => {
        setHelpText("Загрузка списка нод...");
        try {
            const payload = await fetchNodeCatalog();
            catalogByGroup.clear();
            const groups = payload?.groups || [];
            renderComfyAlert(payload?.comfyui || null);
            fillGroupSelect(groups);
            const summary = groups
                .map((group) => {
                    const label = GROUP_LABELS[group.id] || group.title || group.id;
                    return `${label}=${group.count}`;
                })
                .join(", ");
            setHelpText(`Группы: ${summary}.`);
        } catch (err) {
            setHelpText(`Ошибка загрузки: ${String(err)}`);
            comfyAlert.style.display = "none";
            comfyAlert.textContent = "";
            groupSelect.innerHTML = "";
            nodeSelect.innerHTML = "";
            moduleInfo.innerHTML = "";
            nodeList.innerHTML = "";
        }
    };

    groupSelect.onchange = () => fillModuleSelect();
    moduleFilter.oninput = () => fillModuleSelect();
    nodeSelect.onchange = () => {
        renderNodeList();
        loadModuleInfo();
    };
    refreshBtn.onclick = async () => {
        refreshBtn.disabled = true;
        setRefreshLine("Обновление статусов модулей: запуск...", "neutral");
        try {
            await refreshModuleRuntimeState();
            const ok = await pollRefreshProgress();
            if (!ok) {
                setRefreshLine("Обновление статусов модулей: завершилось с ошибкой.", "warn");
            }
        } catch (err) {
            setRefreshLine(`Обновление статусов модулей: ошибка (${String(err)}).`, "warn");
        } finally {
            refreshBtn.disabled = false;
        }
        await loadCatalog();
    };

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
