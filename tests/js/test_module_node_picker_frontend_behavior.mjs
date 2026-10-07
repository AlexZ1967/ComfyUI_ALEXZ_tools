/**
 * Module: tests/js/test_module_node_picker_frontend_behavior.mjs
 * Author: AlexZ1967
 * Last updated: 2026-03-06
 *
 * Description:
 *   Lightweight frontend behavioral checks for Module Node Picker flows.
 *
 * Purpose:
 *   Verifies tab-relay state transitions and refresh/update progress handling
 *   in pure JS modules without browser runtime.
 */

import assert from "node:assert/strict";

import {
    centerNodeInCanvas,
    getCanvasCenterInsertPos,
} from "../../web/ui/module_node_picker_node_factory.js";
import { createBusyUiController } from "../../web/orchestration/ui/module_node_picker_busy_ui.js";
import {
    clearLegacyPersistentFlags,
    createRuntimeStatusAccessors,
    getRuntimePickerState,
    loadComfyCheckMode,
    saveComfyCheckMode,
} from "../../web/state/module_node_picker_runtime_state.js";
import {
    clearModuleNodePickerRelayState,
    getModuleNodePickerRelayState,
    setModuleNodePickerRelayState,
} from "../../web/orchestration/relay/module_node_picker_tab_relay_state.js";
import { createModuleNodePickerPollingController } from "../../web/orchestration/flow/progress/module_node_picker_polling_controller.js";
import { runRefreshCustomNodesInfoAction } from "../../web/orchestration/flow/actions/module_node_picker_actions.js";
import {
    maybeInstallChangedRequirementsFlow,
    pollRefreshProgressLoop,
    pollUpdateProgressLoop,
} from "../../web/orchestration/flow/progress/module_node_picker_update_flow.js";
import { renderComfyAlertCard } from "../../web/ui/module_node_picker_alerts.js";
import { createModuleNodePickerRoot } from "../../web/ui/module_node_picker_layout.js";
import { createRelayDomOwnershipController } from "../../web/orchestration/relay/module_node_picker_tab_relay_dom_ownership.js";
import {
    createIsolatedSidebarRenderer,
    installCustomSidebarHostIsolation,
} from "../../web/orchestration/core/infra/module_node_picker_sidebar_hosts.js";

async function testCustomSidebarMountIsolation() {
    const documentObj = {
        createElement() {
            return {
                ownerDocument: documentObj,
                children: [], parentElement: null, dataset: {}, style: {},
                get childElementCount() { return this.children.length; },
                get firstElementChild() { return this.children[0] || null; },
                appendChild(child) {
                    if (child.parentElement) {
                        const siblings = child.parentElement.children;
                        siblings.splice(siblings.indexOf(child), 1);
                    }
                    this.children.push(child);
                    child.parentElement = this;
                    return child;
                },
                replaceChildren(...children) {
                    for (const child of this.children) child.parentElement = null;
                    this.children = [];
                    for (const child of children) this.appendChild(child);
                },
            };
        },
    };
    const shared = documentObj.createElement();
    const receiver = {};
    const token = {};
    const mapMounts = new Set();
    const mapRender = createIsolatedSidebarRenderer(function (mount, argument) {
        assert.equal(this, receiver);
        assert.equal(argument, token);
        mapMounts.add(mount);
        if (!mount._vnode) mount._vnode = { el: mount.appendChild(documentObj.createElement()) };
        assert.equal(mount._vnode.el.parentElement, mount, "Vue DOM must survive other renderers");
        return argument;
    }, "nodes-map");
    const pngElement = documentObj.createElement();
    const pngRender = createIsolatedSidebarRenderer((mount) => mount.appendChild(pngElement), "png-info");
    const pickerRender = createIsolatedSidebarRenderer((mount) => mount.replaceChildren(documentObj.createElement()), "picker");
    const showMap = (host) => {
        assert.equal(mapRender.call(receiver, host, token), token);
        assert.equal(host.childElementCount, 1);
        assert.equal(host.firstElementChild.dataset.alexzSidebarTab, "nodes-map");
        assert.equal(pngElement.parentElement?.parentElement || null, null, "PNG Info must not overlap NodesMap");
    };
    for (let cycle = 0; cycle < 3; cycle += 1) {
        showMap(shared);
        const mapMount = shared.firstElementChild;
        pngRender(shared);
        assert.equal(mapMount.parentElement, null);
        assert.equal(shared.firstElementChild.dataset.alexzSidebarTab, "png-info");
        showMap(shared);
        pickerRender(shared);
        assert.equal(mapMount.parentElement, null);
        showMap(shared);
    }
    assert.equal(mapMounts.size, 1, "reuse renderer mount across custom-tab switches");
    // После Apps ComfyUI создаёт новый внешний host, без чужого renderer state.
    showMap(documentObj.createElement());
    assert.equal(mapMounts.size, 2);
    assert.equal(shared._vnode, undefined);

    let cleanupCalls = 0;
    let doctorMount;
    const doctorTab = {
        id: "doctor", type: "custom",
        render(mount) {
            doctorMount = mount;
            shared.style.minWidth = "560px";
            mount.replaceChildren(documentObj.createElement());
        },
        destroy() {
            assert.equal(this, doctorTab, "preserve cleanup callback receiver");
            assert.equal(doctorMount.parentElement, shared, "cleanup must precede incoming mount");
            cleanupCalls += 1;
            doctorMount.replaceChildren();
            shared.style.minWidth = "";
        },
    };
    let sidebarSubscription;
    const sidebar = {
        sidebarTabs: [doctorTab], activeSidebarTabId: "doctor",
        $subscribe(callback) { sidebarSubscription = callback; },
    };
    installCustomSidebarHostIsolation({ extensionManager: { sidebarTab: sidebar } });
    for (let cycle = 0; cycle < 2; cycle += 1) {
        doctorTab.render(shared);
        doctorTab.render(shared);
        assert.equal(cleanupCalls, cycle, "do not deactivate when rerendering the same tab");
        assert.equal(shared.style.minWidth, "560px");
        showMap(shared);
        assert.equal(cleanupCalls, cycle + 1);
        assert.equal(shared.style.minWidth, "", "restore layout on custom-tab takeover");
        assert.equal(doctorMount.childElementCount, 0);
        pngRender(shared);
        assert.equal(cleanupCalls, cycle + 1, "do not repeat cleanup after takeover");
    }
    for (const nextId of ["apps", undefined]) {
        sidebar.activeSidebarTabId = "doctor";
        sidebarSubscription();
        doctorTab.render(shared);
        const before = cleanupCalls;
        sidebar.activeSidebarTabId = nextId;
        sidebarSubscription();
        assert.equal(cleanupCalls, before + 1, "cleanup on Vue transition or sidebar closing");
        assert.equal(shared.style.minWidth, "");
        doctorTab.destroy();
        assert.equal(cleanupCalls, before + 1, "ComfyUI unmount must not duplicate cleanup");
        showMap(shared);
        assert.equal(cleanupCalls, before + 1, "incoming custom mount must not duplicate cleanup");
    }
}

async function testSidebarIsolationRegistrationLifecycle() {
    let subscription;
    let subscriptions = 0;
    const originalRender = () => {};
    const sidebar = {
        sidebarTabs: [
            { id: "custom", type: "custom", render: originalRender },
            { id: "apps", type: "vue", render: originalRender },
        ],
        $subscribe(callback, options) {
            subscription = callback;
            subscriptions += 1;
            assert.equal(options.flush, "sync");
            assert.equal(options.detached, true);
        },
    };
    const app = { extensionManager: { sidebarTab: sidebar } };
    const sync = installCustomSidebarHostIsolation(app);
    const wrapped = sidebar.sidebarTabs[0].render;
    assert.notEqual(wrapped, originalRender);
    assert.equal(sidebar.sidebarTabs[1].render, originalRender, "leave Vue panels unchanged");
    assert.equal(installCustomSidebarHostIsolation(app), sync);
    assert.equal(subscriptions, 1);
    assert.equal(sidebar.sidebarTabs[0].render, wrapped, "do not stack wrappers");
    const lateTab = { id: "late-easyuse", type: "custom", render: originalRender };
    sidebar.sidebarTabs = [...sidebar.sidebarTabs, lateTab];
    subscription();
    assert.notEqual(lateTab.render, originalRender, "isolate late sidebar registrations");
    const lateWrapped = lateTab.render;
    subscription();
    assert.equal(lateTab.render, lateWrapped);
}

async function testSharedSidebarPreservesForeignRenderer() {
    const previousElement = globalThis.Element;
    const previousDocument = globalThis.document;
    class SidebarElement {
        constructor() {
            this.children = [];
            this.parentElement = null;
            this.className = "";
            this.connected = false;
            const properties = new Map();
            this.style = {
                getPropertyValue: (name) => properties.get(name)?.value || "",
                getPropertyPriority: (name) => properties.get(name)?.priority || "",
                setProperty: (name, value, priority = "") => properties.set(name, { value, priority }),
                removeProperty: (name) => properties.delete(name),
            };
            this.classList = { contains: (name) => this.className.split(/\s+/).includes(name) };
        }
        get isConnected() {
            return this.connected || Boolean(this.parentElement?.isConnected);
        }
        appendChild(child) {
            child.remove();
            this.children.push(child);
            child.parentElement = this;
            return child;
        }
        removeChild(child) {
            this.children.splice(this.children.indexOf(child), 1);
            child.parentElement = null;
        }
        remove() {
            this.parentElement?.removeChild(this);
        }
    }
    globalThis.Element = SidebarElement;
    globalThis.document = { createElement: () => new SidebarElement() };
    try {
        const host = new SidebarElement();
        host.connected = true;
        const foreign = host.appendChild(new SidebarElement());
        foreign.style.setProperty("display", "flex");
        const alreadyHidden = host.appendChild(new SidebarElement());
        alreadyHidden.style.setProperty("display", "none", "important");
        const rendererState = { el: foreign };
        host._vnode = rendererState;

        // Vue должен сохранить тот же живой DOM после каждого custom-tab round trip.
        for (let cycle = 0; cycle < 3; cycle += 1) {
            const root = createModuleNodePickerRoot(host);
            const ownership = createRelayDomOwnershipController({ root, mountHost: host });
            ownership.ensureAttached();
            ownership.ensureAttached();
            assert.equal(foreign.isConnected, true, "foreign renderer DOM must stay connected");
            assert.equal(host._vnode, rendererState, "foreign renderer state must remain untouched");
            assert.equal(foreign.style.getPropertyValue("display"), "none");
            assert.equal(foreign.style.getPropertyPriority("display"), "important");
            ownership.ensureDetached();
            assert.equal(foreign.style.getPropertyValue("display"), "flex");
            assert.equal(foreign.style.getPropertyPriority("display"), "");
            assert.equal(alreadyHidden.style.getPropertyValue("display"), "none");
            assert.equal(alreadyHidden.style.getPropertyPriority("display"), "important");
            assert.equal(rendererState.el.parentElement, host);
            assert.equal(root.isConnected, false);
        }

        const staleRoot = createModuleNodePickerRoot(host);
        const root = createModuleNodePickerRoot(host);
        assert.equal(staleRoot.parentElement, null, "replace only the previous picker root");
        assert.equal(foreign.isConnected, true);
        const ownership = createRelayDomOwnershipController({ root, mountHost: host });
        ownership.ensureAttached();
        // ComfyUI может отключить host целиком при переходе custom -> Apps.
        host.connected = false;
        ownership.ensureDetached();
        assert.equal(foreign.style.getPropertyValue("display"), "flex", "restore disconnected host styles");

        const nextHost = new SidebarElement();
        nextHost.connected = true;
        const nextForeign = nextHost.appendChild(new SidebarElement());
        nextHost.appendChild(root);
        ownership.ensureAttached();
        assert.equal(nextForeign.style.getPropertyValue("display"), "none");
        ownership.ensureDetached();
        assert.equal(nextForeign.style.getPropertyValue("display"), "", "restore absent inline display");
        assert.equal(nextForeign.isConnected, true);
    } finally {
        globalThis.Element = previousElement;
        globalThis.document = previousDocument;
    }
}

function makeClassList() {
    const names = new Set();
    return {
        add: (...items) => {
            for (const item of items) {
                names.add(String(item));
            }
        },
        remove: (...items) => {
            for (const item of items) {
                names.delete(String(item));
            }
        },
        contains: (name) => names.has(String(name)),
    };
}

async function testRuntimeStateAccessors() {
    const mem = new Map();
    const windowObj = {
        localStorage: {
            getItem: (key) => mem.get(String(key)) || null,
            setItem: (key, value) => mem.set(String(key), String(value)),
            removeItem: (key) => mem.delete(String(key)),
        },
    };

    const stateA = getRuntimePickerState(windowObj, "__runtime__");
    const stateB = getRuntimePickerState(windowObj, "__runtime__");
    assert.equal(stateA, stateB, "runtime state must be stable per key");

    const accessors = createRuntimeStatusAccessors(stateA);
    accessors.setPendingCustomRefresh(true);
    accessors.setPendingUpdate(true);
    accessors.setPendingComfyInfoRefresh(true);
    assert.equal(accessors.hasPendingCustomRefresh(), true);
    assert.equal(accessors.hasPendingUpdate(), true);
    assert.equal(accessors.hasPendingComfyInfoRefresh(), true);
    accessors.clearPendingCustomRefresh();
    accessors.clearPendingUpdate();
    accessors.clearPendingComfyInfoRefresh();
    assert.equal(accessors.hasPendingCustomRefresh(), false);
    assert.equal(accessors.hasPendingUpdate(), false);
    assert.equal(accessors.hasPendingComfyInfoRefresh(), false);

    saveComfyCheckMode(windowObj, "mode_key", "COMMITS");
    assert.equal(loadComfyCheckMode(windowObj, "mode_key"), "commits");
    saveComfyCheckMode(windowObj, "mode_key", "other");
    assert.equal(loadComfyCheckMode(windowObj, "mode_key"), "releases");

    mem.set("legacy_custom", "1");
    mem.set("legacy_pending_refresh", "1");
    mem.set("legacy_pending_update", "1");
    clearLegacyPersistentFlags(windowObj, {
        customStatusCheckedKey: "legacy_custom",
        pendingCustomRefreshKey: "legacy_pending_refresh",
        pendingUpdateKey: "legacy_pending_update",
    });
    assert.equal(mem.has("legacy_custom"), false);
    assert.equal(mem.has("legacy_pending_refresh"), false);
    assert.equal(mem.has("legacy_pending_update"), false);
}

async function testRelayStateTransitions() {
    globalThis.window = {};
    setModuleNodePickerRelayState({ bindToken: "t1", active: true });
    assert.deepEqual(getModuleNodePickerRelayState(), { bindToken: "t1", active: true });
    clearModuleNodePickerRelayState();
    assert.equal(getModuleNodePickerRelayState(), null);
}

async function testRefreshProgressLoopBehavior() {
    const lines = [];
    const customAlert = { style: {}, classList: makeClassList() };
    const customAlertText = { textContent: "" };
    const statuses = [
        { refresh: { running: true, phase: "scan", message: "phase-1" } },
        { refresh: { running: false, phase: "done", message: "phase-2" } },
    ];

    const ok = await pollRefreshProgressLoop({
        shouldContinue: () => true,
        isTokenActive: () => true,
        fetchModuleRefreshStatus: async () => statuses.shift() || { refresh: { running: false, phase: "done" } },
        formatRefreshLine: (refresh) => ({
            text: String(refresh?.message || ""),
            tone: refresh?.running ? "warn" : "ok",
        }),
        setRefreshLine: (text, tone) => lines.push({ text, tone }),
        getProcessTarget: () => "custom",
        customAlert,
        customAlertText,
        sleepMs: 1,
    });

    assert.equal(ok, true);
    assert.equal(lines.length >= 2, true);
    assert.equal(customAlert.style.display, "block");
    assert.equal(customAlertText.textContent, "phase-2");
    assert.equal(customAlert.classList.contains("alexz-mod-picker-status-card--ok"), true);
}

async function testUpdateProgressLoopBehavior() {
    const lines = [];
    const statuses = [
        { update: { running: true, phase: "update", message: "u1" } },
        { update: { running: false, phase: "done", message: "u2", updated: 1 } },
    ];
    const result = await pollUpdateProgressLoop({
        shouldContinue: () => true,
        isTokenActive: () => true,
        fetchModuleUpdateStatus: async () => statuses.shift() || { update: { running: false, phase: "done" } },
        formatUpdateLine: (update) => ({
            text: String(update?.message || ""),
            tone: update?.running ? "neutral" : "ok",
        }),
        setRefreshLine: (text, tone) => lines.push({ text, tone }),
        sleepMs: 1,
    });

    assert.equal(lines.length >= 2, true);
    assert.equal(result?.phase, "done");
    assert.equal(result?.updated, 1);
}

async function testPollingControllerInvalidation() {
    const controller = createModuleNodePickerPollingController({
        shouldContinue: () => true,
        fetchModuleRefreshStatus: async () => ({ refresh: { running: true, phase: "scan", message: "loop" } }),
        formatRefreshLine: (refresh) => ({ text: String(refresh?.message || ""), tone: "neutral" }),
        setRefreshLine: () => {},
        refreshSleepMs: 15,
    });

    const pending = controller.pollRefreshProgress();
    setTimeout(() => controller.invalidate(), 0);
    const result = await pending;
    assert.equal(result, false, "invalidate must stop active polling loop");
}

async function testCanvasCenterPlacement() {
    const app = {
        canvas: {
            canvas: {
                width: 800,
                height: 600,
            },
            ds: {
                scale: 2,
                offset: [-100, -200],
                visible_area: [100, 200, 400, 300],
            },
        },
    };
    const center = getCanvasCenterInsertPos(app);
    assert.deepEqual(center, [300, 350]);

    const node = {
        size: [120, 80],
    };
    centerNodeInCanvas(node, app);
    assert.deepEqual(node.pos, [240, 310]);
}

async function testCustomRefreshFlowFinalizesBusyState() {
    let busySetCalls = 0;
    let resetBusyCalls = 0;
    let syncUpdateCalls = 0;
    let clearPendingCalls = 0;
    let clearUpdatedSessionCalls = 0;
    let refreshOptions = null;

    await runRefreshCustomNodesInfoAction({
        shouldContinue: () => true,
        setCustomStatusChecked: () => {},
        setPendingCustomRefresh: () => {},
        clearPendingCustomRefresh: () => {
            clearPendingCalls += 1;
        },
        setActionBusy: () => {
            busySetCalls += 1;
        },
        resetBusyState: () => {
            resetBusyCalls += 1;
        },
        syncUpdateAllButton: () => {
            syncUpdateCalls += 1;
        },
        clearUpdatedModulesSession: () => {
            clearUpdatedSessionCalls += 1;
        },
        setProcessTarget: () => {},
        setProcessAction: () => {},
        setRefreshLine: () => {},
        syncUpstreams: false,
        refreshModuleRuntimeState: async (options) => { refreshOptions = options; return {}; },
        pollRefreshProgress: async () => true,
        acknowledgeAllModuleNovelty: async () => ({}),
        loadCatalog: async () => ({ ok: true }),
    });

    assert.equal(busySetCalls, 1);
    assert.equal(resetBusyCalls, 2);
    assert.equal(syncUpdateCalls, 2);
    assert.equal(clearPendingCalls, 1);
    assert.equal(clearUpdatedSessionCalls, 1);
    assert.equal(refreshOptions.syncUpstreams, false);
}

async function testBusyUiForceResetBypassesLifecycleGuard() {
    const controls = {
        refreshBtn: { disabled: false },
        comfyInfoBtn: { disabled: false },
        comfyModeSelect: { disabled: false },
        categorySelect: { disabled: false },
        groupSelect: { disabled: false, options: [] },
        nodeSelect: { disabled: false, options: [] },
        moduleFilter: { disabled: false },
        moduleInfo: {
            style: {},
            querySelectorAll: () => [],
        },
        nodeList: {
            style: {},
        },
    };
    let alive = true;
    const busyUi = createBusyUiController({
        shouldContinue: () => alive,
        controls,
        getProcessUi: () => null,
    });

    busyUi.setActionBusy(true);
    assert.equal(controls.refreshBtn.disabled, true);
    alive = false;
    busyUi.resetBusyState(true);

    assert.equal(controls.refreshBtn.disabled, false);
    assert.equal(controls.comfyInfoBtn.disabled, false);
    assert.equal(controls.categorySelect.disabled, false);
    assert.equal(controls.moduleInfo.style.pointerEvents, "");
    assert.equal(controls.nodeList.style.pointerEvents, "");
}

async function testRequirementsFollowupUsesManualAdvisoryText() {
    const refreshLines = [];
    const actions = [];

    await maybeInstallChangedRequirementsFlow(
        {
            scope: "comfyui",
            requirements_changed: true,
            requirements_paths: ["/tmp/ComfyUI/requirements.txt"],
        },
        {
            shouldContinue: () => true,
            setRefreshLine: (text, tone) => refreshLines.push({ text, tone }),
            setProcessAction: (label, btnText, onClick) => actions.push({ label, btnText, onClickType: typeof onClick }),
        }
    );

    assert.equal(refreshLines.at(-1)?.tone, "warn");
    assert.equal(refreshLines.at(-1)?.text.includes("Manual dependency install required"), true);
    assert.equal(actions.length, 1);
    assert.equal(actions[0].label.includes('python -m pip install -r "/tmp/ComfyUI/requirements.txt"'), true);
    assert.equal(actions[0].btnText, "");
    assert.equal(actions[0].onClickType, "object");
}

async function testComfyReleaseCheckDegradedUsesNeutralWarningText() {
    const comfyAlert = {
        style: {},
        classList: makeClassList(),
    };
    const comfyAlertText = { textContent: "" };

    renderComfyAlertCard({
        info: {
            check_mode: "releases",
            update_status: "unknown",
            release_check_degraded: true,
            release_check_reason: "release_tag_not_resolved",
            release_tag: "v0.3.40",
        },
        comfyMode: "releases",
        comfyAlert,
        comfyAlertText,
    });

    assert.equal(comfyAlert.style.display, "block");
    assert.equal(comfyAlert.classList.contains("alexz-mod-picker-status-card--neutral"), true);
    assert.equal(
        comfyAlertText.textContent.includes("could not resolve release tag v0.3.40 locally"),
        true
    );
}

async function main() {
    const tests = [
        ["custom sidebar mount isolation", testCustomSidebarMountIsolation],
        ["sidebar isolation registration lifecycle", testSidebarIsolationRegistrationLifecycle],
        ["shared sidebar preserves foreign renderer", testSharedSidebarPreservesForeignRenderer],
        ["runtime state accessors", testRuntimeStateAccessors],
        ["relay state transitions", testRelayStateTransitions],
        ["refresh progress loop behavior", testRefreshProgressLoopBehavior],
        ["update progress loop behavior", testUpdateProgressLoopBehavior],
        ["polling invalidation", testPollingControllerInvalidation],
        ["canvas center placement", testCanvasCenterPlacement],
        ["custom refresh finalizes busy state", testCustomRefreshFlowFinalizesBusyState],
        ["busy ui force reset bypasses lifecycle guard", testBusyUiForceResetBypassesLifecycleGuard],
        ["requirements follow-up uses manual advisory", testRequirementsFollowupUsesManualAdvisoryText],
        ["comfy release-check degraded text", testComfyReleaseCheckDegradedUsesNeutralWarningText],
    ];

    for (const [name, fn] of tests) {
        await fn();
        // Keep output compact and CI-readable.
        console.log(`ok: ${name}`);
    }
}

main().catch((err) => {
    console.error("frontend behavior test failed:", err);
    process.exit(1);
});
