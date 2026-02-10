/**
 * Module Node Picker tab relay helper.
 *
 * Provides sidebar-tab visibility synchronization helpers that keep the picker
 * panel mounted/unmounted correctly across ComfyUI tab switches.
 */

const TAB_RELAY_STATE_KEY = "__alexz_module_picker_tab_relay_state_v2__";

function getSidebarApi(app) {
    const manager = app?.extensionManager;
    return manager?.sidebarTab || manager || null;
}

function getActiveSidebarTabId(app) {
    const sidebar = getSidebarApi(app);
    if (!sidebar) {
        return "";
    }
    const active = sidebar.activeSidebarTabId ?? sidebar.activeSidebarTab ?? "";
    return String(active || "");
}

function activateSidebarTab(app, tabId) {
    if (!tabId) {
        return false;
    }
    const sidebar = getSidebarApi(app);
    if (!sidebar || typeof sidebar.activateSidebarTab !== "function") {
        return false;
    }
    try {
        sidebar.activateSidebarTab(tabId);
        return true;
    } catch (_err) {
        return false;
    }
}

function forceActivateSidebarTab(app, tabId, ownSidebarTabId = "") {
    if (!tabId) {
        return false;
    }
    // Safety mode: avoid hard-forcing sidebar internals to prevent tab lockups.
    return activateSidebarTab(app, tabId);
}

function extractTabIdFromButton(buttonEl) {
    if (!(buttonEl instanceof Element)) {
        return "";
    }
    const idAttr = String(buttonEl.getAttribute("id") || "").trim();
    if (idAttr.endsWith("-tab-button")) {
        return idAttr.slice(0, -"-tab-button".length);
    }
    const dataTabId = String(
        buttonEl.getAttribute("data-tab-id")
        || buttonEl.getAttribute("data-sidebar-tab")
        || buttonEl.getAttribute("data-tab")
        || ""
    ).trim();
    if (dataTabId) {
        return dataTabId;
    }
    for (const cls of Array.from(buttonEl.classList || [])) {
        if (cls.endsWith("-tab-button")) {
            return cls.slice(0, -"-tab-button".length);
        }
    }
    return "";
}

function resolveSidebarButtonFromEvent(event) {
    const direct = event?.target;
    if (direct instanceof Element) {
        const closest = direct.closest(
            ".side-bar-button, [class*='-tab-button'], [role='tab'], button[aria-label], button[title]"
        );
        if (closest) {
            return closest;
        }
    }
    if (typeof event?.composedPath === "function") {
        for (const item of event.composedPath()) {
            if (!(item instanceof Element)) {
                continue;
            }
            if (item.classList?.contains("side-bar-button")) {
                return item;
            }
            if (extractTabIdFromButton(item)) {
                return item;
            }
            if (String(item.getAttribute("role") || "").toLowerCase() === "tab") {
                return item;
            }
        }
    }
    return null;
}

function isSidebarContextEvent(event) {
    const direct = event?.target;
    if (direct instanceof Element) {
        const sidebarMatch = direct.closest(
            ".side-bar, .sidebar, .comfy-sidebar, [class*='sidebar'], [class*='side-bar']"
        );
        if (sidebarMatch) {
            return true;
        }
    }
    if (typeof event?.composedPath === "function") {
        for (const item of event.composedPath()) {
            if (!(item instanceof Element)) {
                continue;
            }
            const cls = String(item.className || "").toLowerCase();
            const id = String(item.id || "").toLowerCase();
            if (
                cls.includes("sidebar")
                || cls.includes("side-bar")
                || id.includes("sidebar")
                || id.includes("side-bar")
            ) {
                return true;
            }
        }
    }
    return false;
}

function isOwnButtonSelected(sidebarTabId) {
    const ownBtn = document.querySelector(`.${sidebarTabId}-tab-button`);
    if (!ownBtn) {
        return null;
    }
    return ownBtn.classList.contains("side-bar-button-selected");
}

function collectSidebarTabDescriptors(app) {
    const sidebar = getSidebarApi(app);
    const pools = [
        sidebar?.tabs,
        sidebar?.sidebarTabs,
        sidebar?.tabConfigs,
        app?.extensionManager?.tabs,
        app?.extensionManager?.sidebarTabs,
    ];
    const out = [];
    for (const pool of pools) {
        if (!Array.isArray(pool)) {
            continue;
        }
        for (const item of pool) {
            if (!item || typeof item !== "object") {
                continue;
            }
            const id = String(item.id || item.tabId || item.key || "").trim();
            const title = String(item.title || item.label || item.tooltip || "").trim();
            if (id) {
                out.push({ id, title });
            }
        }
    }
    return out;
}

function hasSidebarTabId(app, tabId) {
    if (!tabId) {
        return false;
    }
    const descriptors = collectSidebarTabDescriptors(app);
    return descriptors.some((x) => String(x.id || "") === String(tabId));
}

function inferFallbackTabIdFromContext(app, event, sidebarTabId) {
    const knownNodesMapId = "easyuse_nodes_map";
    const descriptors = collectSidebarTabDescriptors(app);
    const composed = typeof event?.composedPath === "function" ? event.composedPath() : [];
    const chunks = [];
    for (const item of composed) {
        if (!(item instanceof Element)) {
            continue;
        }
        const cls = String(item.className || "").toLowerCase();
        const text = String(item.textContent || "").toLowerCase().trim();
        const title = String(item.getAttribute("title") || item.getAttribute("aria-label") || "").toLowerCase().trim();
        if (cls) {
            chunks.push(cls);
        }
        if (text) {
            chunks.push(text);
        }
        if (title) {
            chunks.push(title);
        }
    }
    const hay = chunks.join(" | ");
    if (hay.includes("nodesmap") || hay.includes("nodes map")) {
        if (hasSidebarTabId(app, knownNodesMapId)) {
            return knownNodesMapId;
        }
    }
    if (hay.includes("pi-sitemap") && hasSidebarTabId(app, knownNodesMapId)) {
        return knownNodesMapId;
    }
    // Generic fallback by title match from descriptors.
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (!id || id === sidebarTabId) {
            continue;
        }
        if (title.includes("nodesmap") || title.includes("nodes map")) {
            return id;
        }
    }
    return "";
}

function inferTabIdFromButton(app, button) {
    const explicit = extractTabIdFromButton(button);
    if (explicit) {
        return explicit;
    }
    const label = String(
        button.getAttribute("title")
        || button.getAttribute("aria-label")
        || button.textContent
        || ""
    ).trim().toLowerCase();
    if (!label) {
        return "";
    }
    const descriptors = collectSidebarTabDescriptors(app);
    for (const item of descriptors) {
        const id = String(item.id || "");
        const title = String(item.title || "").toLowerCase();
        if (title && (title === label || title.includes(label) || label.includes(title))) {
            return id;
        }
    }
    if (label.includes("nodesmap") || label.includes("nodes map")) {
        // Known EasyUse tab id variants seen in different builds.
        return "easyuse_nodes_map";
    }
    return "";
}

function getContainerState(root) {
    const container = root?.parentElement;
    if (!(container instanceof Element)) {
        return { childCount: 0, childShort: "n/a" };
    }
    const out = [];
    let childCount = 0;
    for (const node of container.childNodes) {
        childCount += 1;
        if (node === root) {
            out.push("ROOT");
            continue;
        }
        if (node.nodeType === Node.TEXT_NODE) {
            const txt = String(node.textContent || "").trim();
            out.push(txt ? `TXT:${txt.slice(0, 20)}` : "TXT:blank");
            continue;
        }
        if (node instanceof Element) {
            const cls = String(node.className || "").trim();
            out.push(cls ? `${node.nodeName}.${cls.split(/\s+/).slice(0, 2).join(".")}` : node.nodeName);
        } else {
            out.push(String(node.nodeName || "NODE"));
        }
    }
    return { childCount, childShort: out.slice(0, 10).join(" | ") || "n/a" };
}

export function unbindModuleNodesTabRelay() {
    const state = window[TAB_RELAY_STATE_KEY];
    if (!state) {
        return;
    }
    if (state.relayTimer) {
        window.clearTimeout(state.relayTimer);
    }
    if (state.tickInterval) {
        window.clearInterval(state.tickInterval);
    }
    if (state.bindButtonsInterval) {
        window.clearInterval(state.bindButtonsInterval);
    }
    if (state.onPointerDown) {
        document.removeEventListener("pointerdown", state.onPointerDown, true);
        window.removeEventListener("pointerdown", state.onPointerDown, true);
    }
    if (state.onMouseDown) {
        document.removeEventListener("mousedown", state.onMouseDown, true);
        window.removeEventListener("mousedown", state.onMouseDown, true);
    }
    if (state.onClick) {
        document.removeEventListener("click", state.onClick, true);
        window.removeEventListener("click", state.onClick, true);
    }
    if (state.onKeyUp) {
        document.removeEventListener("keyup", state.onKeyUp, true);
    }
    if (state.onFocusIn) {
        document.removeEventListener("focusin", state.onFocusIn, true);
    }
    if (Array.isArray(state.boundButtons)) {
        for (const item of state.boundButtons) {
            if (!item?.el || !item?.handler) {
                continue;
            }
            item.el.removeEventListener("pointerdown", item.handler, true);
            item.el.removeEventListener("mousedown", item.handler, true);
            item.el.removeEventListener("click", item.handler, true);
        }
    }
    window[TAB_RELAY_STATE_KEY] = null;
}

export function bindModuleNodesTabRelay({ app, root, sidebarTabId, onDiag }) {
    unbindModuleNodesTabRelay();

    let lastClickedTabId = "";
    let lastClickedAt = 0;
    let lastDiagSig = "";
    const boundButtons = [];
    let relayTimer = 0;
    const homeContainer = root.parentElement instanceof Element ? root.parentElement : null;

    const emitDiag = (reason, clickedTabId = "") => {
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const containerState = getContainerState(root);
        const diag = {
            reason,
            activeTabId: getActiveSidebarTabId(app) || "n/a",
            lastClickedTabId: clickedTabId || lastClickedTabId || "n/a",
            ownBtnFound: ownSelected !== null,
            ownBtnSelected: ownSelected,
            rootDisplay: root.style.display || "",
            childNodesCount: containerState.childCount,
            childNodesShort: containerState.childShort,
        };
        const sig = JSON.stringify(diag);
        if (sig === lastDiagSig) {
            return;
        }
        lastDiagSig = sig;
        onDiag?.(diag);
    };

    const ensureRootAttached = () => {
        if (root.isConnected) {
            return true;
        }
        if (homeContainer && homeContainer.isConnected) {
            homeContainer.appendChild(root);
            return true;
        }
        return false;
    };

    const ensureRootDetached = () => {
        if (!root.isConnected) {
            return true;
        }
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        return true;
    };

    const syncVisibility = (reason, clickedTabId = "") => {
        const activeTabId = getActiveSidebarTabId(app);
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const shouldShow = ownSelected === true || activeTabId === sidebarTabId;
        if (shouldShow) {
            ensureRootAttached();
            root.style.display = "";
        } else {
            ensureRootDetached();
        }
        emitDiag(reason, clickedTabId);
    };

    const maybeForceRecentTab = (reason) => {
        const activeTabId = getActiveSidebarTabId(app);
        const ownSelected = isOwnButtonSelected(sidebarTabId);
        const clickedRecently = Boolean(
            lastClickedTabId &&
            lastClickedTabId !== sidebarTabId &&
            Date.now() - lastClickedAt < 1800
        );
        if (!clickedRecently || ownSelected !== true || activeTabId !== sidebarTabId) {
            return false;
        }
        // Safety mode: do not auto-force tab changes from background tick.
        return false;
    };

    const processTabButton = (button) => {
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId) {
            return;
        }
        if (tabId === sidebarTabId) {
            lastClickedTabId = sidebarTabId;
            lastClickedAt = Date.now();
            syncVisibility("relay_own_tab_click", sidebarTabId);
            return;
        }
        lastClickedTabId = tabId;
        lastClickedAt = Date.now();
        if (isOwnButtonSelected(sidebarTabId) !== true) {
            return;
        }
        if (relayTimer) {
            window.clearTimeout(relayTimer);
        }
        relayTimer = window.setTimeout(() => {
            relayTimer = 0;
            const liveState = window[TAB_RELAY_STATE_KEY];
            if (liveState) {
                liveState.relayTimer = 0;
            }
            const activeTabId = getActiveSidebarTabId(app);
            const ownSelected = isOwnButtonSelected(sidebarTabId);
            if (activeTabId === tabId || ownSelected === false) {
                syncVisibility("relay_native_ok", tabId);
                return;
            }
            const forced = forceActivateSidebarTab(app, tabId, sidebarTabId);
            syncVisibility(forced ? "relay_forced" : "relay_force_failed", tabId);
        }, 60);
        const liveState = window[TAB_RELAY_STATE_KEY];
        if (liveState) {
            liveState.relayTimer = relayTimer;
        }
    };

    const handleEvent = (event) => {
        const direct = event?.target;
        if (direct instanceof Element && root.contains(direct)) {
            return;
        }
        const button = resolveSidebarButtonFromEvent(event);
        if (!button) {
            if (isOwnButtonSelected(sidebarTabId) === true && isSidebarContextEvent(event)) {
                const fallbackTabId = inferFallbackTabIdFromContext(app, event, sidebarTabId);
                lastClickedTabId = fallbackTabId || "(unknown-other-tab)";
                lastClickedAt = Date.now();
                syncVisibility("relay_unknown_tab_click", lastClickedTabId);
            }
            return;
        }
        const tabId = inferTabIdFromButton(app, button);
        if (!tabId && isOwnButtonSelected(sidebarTabId) === true) {
            const ownMarker = `${sidebarTabId}-tab-button`;
            const isOwn = button.classList?.contains(ownMarker)
                || String(button.getAttribute("id") || "") === ownMarker;
            if (!isOwn) {
                const fallbackTabId = inferFallbackTabIdFromContext(app, event, sidebarTabId);
                lastClickedTabId = fallbackTabId || "(unknown-other-tab)";
                lastClickedAt = Date.now();
                syncVisibility("relay_unknown_tab_click", lastClickedTabId);
            }
            return;
        }
        processTabButton(button);
    };

    const bindDirectButtonListeners = () => {
        const buttons = Array.from(document.querySelectorAll(".side-bar-button, [class*='-tab-button']"));
        for (const button of buttons) {
            if (!(button instanceof Element)) {
                continue;
            }
            const tabId = inferTabIdFromButton(app, button);
            if (!tabId || tabId === sidebarTabId) {
                continue;
            }
            if (boundButtons.some((x) => x.el === button)) {
                continue;
            }
            const handler = () => processTabButton(button);
            button.addEventListener("pointerdown", handler, true);
            button.addEventListener("mousedown", handler, true);
            button.addEventListener("click", handler, true);
            boundButtons.push({ el: button, handler });
        }
    };

    const onPointerDown = (event) => handleEvent(event);
    const onMouseDown = (event) => handleEvent(event);
    const onClick = (event) => handleEvent(event);
    const onKeyUp = () => syncVisibility("relay_keyup");
    const onFocusIn = (event) => handleEvent(event);

    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("mousedown", onMouseDown, true);
    document.addEventListener("click", onClick, true);
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("click", onClick, true);
    document.addEventListener("keyup", onKeyUp, true);
    document.addEventListener("focusin", onFocusIn, true);

    bindDirectButtonListeners();
    const bindButtonsInterval = window.setInterval(bindDirectButtonListeners, 1000);
    const tickInterval = window.setInterval(() => {
        if (!root.isConnected) {
            return;
        }
        if (!maybeForceRecentTab("relay_tick_forced")) {
            syncVisibility("relay_tick");
        }
    }, 220);

    syncVisibility("relay_init");
    window[TAB_RELAY_STATE_KEY] = {
        relayTimer,
        tickInterval,
        bindButtonsInterval,
        onPointerDown,
        onMouseDown,
        onClick,
        onKeyUp,
        onFocusIn,
        boundButtons,
    };
}
