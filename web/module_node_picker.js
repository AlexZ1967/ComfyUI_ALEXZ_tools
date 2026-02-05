import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXT_NAME = "ALEXZ.Tools.ModuleNodePicker";
const DEFAULT_MODULE = "ComfyUI_ALEXZ_tools";
const SIDEBAR_TAB_ID = "alexz-module-nodes";

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
    .alexz-mod-picker-input {
        width: 100%;
    }
    .alexz-mod-picker-module-select {
        width: 100%;
    }
    .alexz-mod-picker-status {
        font-size: 12px;
        opacity: 0.8;
    }
    .alexz-mod-picker-group {
        border: 1px solid var(--border-color, #444);
        border-radius: 6px;
        padding: 7px;
        margin-bottom: 8px;
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
        opacity: 0.8;
        line-height: 1.25em;
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
    const candidates = [
        nodeInfo.node_name,
        nodeInfo.display_name,
    ].filter(Boolean);
    for (const name of candidates) {
        const node = LiteGraph.createNode(name);
        if (node) {
            return node;
        }
    }
    return null;
}

async function fetchModuleNodes(moduleName) {
    const q = (moduleName || "").trim();
    const resp = await api.fetchApi(
        `/alexz_tools/module_nodes?module=${encodeURIComponent(q)}`,
        { cache: "no-store" }
    );
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
}

async function fetchModuleList() {
    const resp = await api.fetchApi("/alexz_tools/module_list", {
        cache: "no-store",
    });
    if (!resp.ok) {
        throw new Error(`API ${resp.status}`);
    }
    return await resp.json();
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
    title.textContent = "Module Node Picker";
    head.appendChild(title);

    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.textContent = "Обновить";
    head.appendChild(refreshBtn);

    const input = document.createElement("input");
    input.type = "text";
    input.className = "alexz-mod-picker-input";
    input.placeholder = "Введите модуль, например ComfyUI_ALEXZ_tools";
    input.value = DEFAULT_MODULE;
    input.setAttribute("list", "alexz-module-picker-list");
    root.appendChild(input);

    const dataList = document.createElement("datalist");
    dataList.id = "alexz-module-picker-list";
    root.appendChild(dataList);

    const moduleSelect = document.createElement("select");
    moduleSelect.className = "alexz-mod-picker-module-select";
    root.appendChild(moduleSelect);

    const status = document.createElement("div");
    status.className = "alexz-mod-picker-status";
    status.textContent = "Введите имя python-модуля.";
    root.appendChild(status);

    const results = document.createElement("div");
    root.appendChild(results);

    let timer = null;

    const renderResults = (payload) => {
        results.innerHTML = "";
        const groups = payload?.results || [];
        if (!groups.length) {
            status.textContent = payload?.query
                ? "Модуль не найден или в нем нет нод."
                : payload?.hint || "Введите имя модуля.";
            return;
        }
        status.textContent = `Найдено модулей: ${groups.length}`;
        for (const group of groups) {
            const groupEl = document.createElement("div");
            groupEl.className = "alexz-mod-picker-group";

            const titleEl = document.createElement("div");
            titleEl.className = "alexz-mod-picker-group-title";
            titleEl.textContent = `${group.module} (${group.count})`;
            groupEl.appendChild(titleEl);

            for (const nodeInfo of group.nodes || []) {
                const item = document.createElement("button");
                item.type = "button";
                item.className = "alexz-mod-picker-node";
                item.onclick = () => {
                    const node = createNodeByInfo(nodeInfo);
                    if (!node) {
                        status.textContent = `Не удалось создать ноду: ${nodeInfo.display_name}`;
                        return;
                    }
                    app.graph.add(node);
                    centerNode(node);
                    app.canvas?.selectNode?.(node, false);
                    app.graph.setDirtyCanvas(true, true);
                    status.textContent = `Добавлена: ${nodeInfo.display_name}`;
                };

                const nameEl = document.createElement("div");
                nameEl.className = "alexz-mod-picker-node-name";
                nameEl.textContent = nodeInfo.display_name;
                item.appendChild(nameEl);

                const descEl = document.createElement("div");
                descEl.className = "alexz-mod-picker-node-desc";
                const category = nodeInfo.category ? ` [${nodeInfo.category}]` : "";
                descEl.textContent = `${nodeInfo.annotation}${category}`;
                item.appendChild(descEl);

                groupEl.appendChild(item);
            }
            results.appendChild(groupEl);
        }
    };

    const runSearch = async () => {
        const moduleName = input.value.trim();
        if (!moduleName) {
            results.innerHTML = "";
            status.textContent = "Выберите модуль из списка или введите вручную.";
            return;
        }
        status.textContent = "Загрузка...";
        try {
            const payload = await fetchModuleNodes(moduleName);
            renderResults(payload);
        } catch (err) {
            results.innerHTML = "";
            status.textContent = `Ошибка запроса: ${String(err)}`;
        }
    };

    const scheduleSearch = () => {
        if (timer) {
            clearTimeout(timer);
        }
        timer = setTimeout(runSearch, 200);
    };

    input.oninput = scheduleSearch;
    input.onkeydown = (event) => {
        if (event.key === "Enter") {
            runSearch();
        }
    };
    refreshBtn.onclick = runSearch;

    const fillModuleDropdown = async () => {
        try {
            const payload = await fetchModuleList();
            const modules = payload?.modules || [];

            moduleSelect.innerHTML = "";
            const topOption = document.createElement("option");
            topOption.value = "";
            topOption.textContent = "Выберите модуль из загруженных...";
            moduleSelect.appendChild(topOption);

            dataList.innerHTML = "";
            for (const item of modules) {
                const moduleName = item.module;
                const label = `${moduleName} (${item.count})`;

                const opt = document.createElement("option");
                opt.value = moduleName;
                opt.textContent = label;
                moduleSelect.appendChild(opt);

                const dl = document.createElement("option");
                dl.value = moduleName;
                dataList.appendChild(dl);
            }

            const hasDefault = modules.some((x) => x.module === DEFAULT_MODULE);
            if (hasDefault) {
                moduleSelect.value = DEFAULT_MODULE;
                input.value = DEFAULT_MODULE;
            }
            status.textContent = `Загружено модулей: ${modules.length}`;
        } catch (err) {
            status.textContent = `Не удалось загрузить список модулей: ${String(err)}`;
        }
    };

    moduleSelect.onchange = () => {
        const val = moduleSelect.value;
        if (!val) {
            return;
        }
        input.value = val;
        runSearch();
    };

    input.onchange = () => {
        moduleSelect.value = input.value.trim();
    };

    fillModuleDropdown();
    runSearch();
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
    button.title = "Открыть поиск нод по модулю";
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
                tooltip: "Поиск и вставка нод по python-модулю",
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
