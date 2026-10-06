/** Isolate custom sidebar renderers from ComfyUI's shared mount element. */

const installedStores = new WeakMap();
const activeMounts = new WeakMap();
const rendererCleanups = new WeakMap();

/**
 * Give a renderer its own cached mount for each ComfyUI sidebar host.
 */
export function createIsolatedSidebarRenderer(render, tabId, destroy) {
    const mounts = new WeakMap();
    const isolatedRender = function (container, ...args) {
        let mount = mounts.get(container);
        if (!mount) {
            mount = container.ownerDocument.createElement("div");
            mount.className = "alexz-custom-sidebar-host";
            mount.dataset.alexzSidebarTab = String(tabId);
            mount.style.width = "100%";
            mount.style.height = "100%";
            mounts.set(container, mount);
        }
        // Переключаем целые mount-элементы, сохраняя их дочерний DOM и renderer.
        // Общий host не содержит Vue-состояние отдельной custom-панели.
        if (container.childElementCount !== 1 || container.firstElementChild !== mount) {
            const previous = activeMounts.get(container);
            // ExtensionSlot вызывает destroy только при unmount, а custom -> custom
            // переиспользует host. Выполняем cleanup до установки следующей панели.
            if (previous?.mount !== mount && previous?.mount.parentElement === container) {
                previous.destroy?.();
            }
            container.replaceChildren(mount);
        }
        let active = true;
        const cleanup = () => {
            if (!active) return;
            active = false;
            destroy?.();
        };
        activeMounts.set(container, { mount, destroy: cleanup });
        rendererCleanups.set(isolatedRender, cleanup);
        return render.call(this, mount, ...args);
    };
    return isolatedRender;
}

/**
 * Isolate existing and subsequently registered custom tabs using the sidebar store.
 */
export function installCustomSidebarHostIsolation(app) {
    const manager = app?.extensionManager;
    const sidebar = manager?.sidebarTab || manager;
    if (!sidebar || typeof sidebar !== "object") {
        return () => {};
    }
    const installed = installedStores.get(sidebar);
    if (installed) {
        installed();
        return installed;
    }

    const wrapped = new WeakSet();
    let syncing = false;
    let previousActiveId = sidebar.activeSidebarTabId;
    const sync = () => {
        if (syncing) return;
        syncing = true;
        try {
            const tabs = sidebar.sidebarTabs || sidebar.tabs;
            if (!Array.isArray(tabs)) return;
            if (previousActiveId !== sidebar.activeSidebarTabId) {
                const previous = tabs.find((tab) => tab.id === previousActiveId);
                previousActiveId = sidebar.activeSidebarTabId;
                // custom -> Vue и закрытие sidebar тоже требуют cleanup до patch DOM.
                // Последующий destroy от ExtensionSlot не должен повторять этот вызов.
                rendererCleanups.get(previous?.render)?.();
            }
            for (const tab of tabs) {
                if (tab.type !== "custom" || typeof tab.render !== "function" || wrapped.has(tab.render)) {
                    continue;
                }
                const destroy = tab.destroy;
                const render = createIsolatedSidebarRenderer(tab.render, tab.id, () => destroy?.call(tab));
                wrapped.add(render);
                tab.render = render;
                if (typeof destroy === "function") {
                    tab.destroy = () => rendererCleanups.get(render)?.();
                }
            }
        } finally {
            syncing = false;
        }
    };
    installedStores.set(sidebar, sync);
    // Публичная Pinia-подписка охватывает и позднюю регистрацию EasyUse.
    // Vue-панели и registerSidebarTab не подменяются.
    if (typeof sidebar.$subscribe === "function") {
        sidebar.$subscribe(sync, { detached: true, flush: "sync" });
    }
    sync();
    return sync;
}
