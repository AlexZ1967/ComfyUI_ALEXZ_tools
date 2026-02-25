/**
 * Module: web/ui/module_node_picker_node_factory.js
 * Author: AlexZ1967
 * Last updated: 2026-02-11
 *
 * Description:
 *   LiteGraph node creation/placement helpers for Module Node Picker.
 *
 * Purpose:
 *   Keep node insertion primitives isolated from picker composition so
 *   render/orchestration modules can reuse a stable helper API.
 */

/**
 * Place node near the visible canvas center with safe fixed fallback.
 */
export function centerNodeInCanvas(node, app) {
    const rawSize = (() => {
        if (Array.isArray(node?.size) && node.size.length >= 2) {
            return node.size;
        }
        if (typeof node?.computeSize === "function") {
            const computed = node.computeSize();
            if (Array.isArray(computed) && computed.length >= 2) {
                return computed;
            }
        }
        if (Number.isFinite(node?.width) && Number.isFinite(node?.height)) {
            return [node.width, node.height];
        }
        return [220, 120];
    })();
    const width = Number.isFinite(rawSize?.[0]) ? Number(rawSize[0]) : 220;
    const height = Number.isFinite(rawSize?.[1]) ? Number(rawSize[1]) : 120;

    const area = app?.canvas?.visible_area;
    if (area && area.length >= 4) {
        let isNewMenuMode = false;
        try {
            const mode = app?.ui?.settings?.getSettingValue?.("Comfy.UseNewMenu", "Disabled");
            isNewMenuMode = mode !== "Disabled" && mode !== false && mode !== null;
        } catch (_err) {
            isNewMenuMode = false;
        }
        const x = Number(area[0]) + Number(area[2]) * 0.5 - (isNewMenuMode ? 0 : width * 0.5);
        const y = Number(area[1]) + Number(area[3]) * 0.5 - (isNewMenuMode ? 0 : height * 0.5);
        node.pos = [
            Number.isFinite(x) ? x : 200,
            Number.isFinite(y) ? y : 120,
        ];
        return;
    }
    node.pos = [200, 120];
}

function safeNumber(value, fallback = 0) {
    return Number.isFinite(value) ? Number(value) : fallback;
}

/**
 * Compute a stable graph-space insertion point near visible canvas center.
 * Mirrors native/manager placement approach for Node 2.0 compatibility.
 */
export function getCanvasCenterInsertPos(app) {
    const dsArea = app?.canvas?.ds?.visible_area;
    if (Array.isArray(dsArea) && dsArea.length >= 4) {
        const dpi = Math.max(Number(globalThis?.devicePixelRatio || 1), 1);
        const x = safeNumber(dsArea[0]);
        const y = safeNumber(dsArea[1]);
        const w = safeNumber(dsArea[2]);
        const h = safeNumber(dsArea[3]);
        return [
            x + (w / dpi) * 0.5,
            y + (h / dpi) * 0.5,
        ];
    }
    const area = app?.canvas?.visible_area;
    if (Array.isArray(area) && area.length >= 4) {
        const x = safeNumber(area[0]);
        const y = safeNumber(area[1]);
        const w = safeNumber(area[2]);
        const h = safeNumber(area[3]);
        return [
            x + w * 0.5,
            y + h * 0.5,
        ];
    }
    return [200, 120];
}

/**
 * Insert node into current graph with Node 1.x/2.0 compatibility fallbacks.
 */
function isNodeInGraph(node, graph) {
    if (!node || !graph) {
        return false;
    }
    if (node.graph === graph) {
        return true;
    }
    if (Array.isArray(graph._nodes) && graph._nodes.includes(node)) {
        return true;
    }
    if (typeof graph.getNodeById === "function" && Number.isFinite(node.id)) {
        return graph.getNodeById(node.id) === node;
    }
    return false;
}

async function callGraphAdd(graph, methodName, node, extraArg) {
    if (typeof graph?.[methodName] !== "function") {
        return false;
    }
    const result = extraArg === undefined
        ? graph[methodName](node)
        : graph[methodName](node, extraArg);
    if (result && typeof result.then === "function") {
        await result;
    }
    return true;
}

function normalizeNodeSizeAfterInsert(node) {
    if (!node) {
        return;
    }
    const computed = typeof node.computeSize === "function" ? node.computeSize() : null;
    if (!Array.isArray(computed) || computed.length < 2) {
        return;
    }
    const width = Number.isFinite(computed[0]) ? Number(computed[0]) : null;
    const height = Number.isFinite(computed[1]) ? Number(computed[1]) : null;
    if (!width || !height) {
        return;
    }
    if (typeof node.setSize === "function") {
        node.setSize([width, height]);
    } else {
        node.size = [width, height];
    }
    if (typeof node.onResize === "function") {
        node.onResize(node.size);
    }
}

export async function addNodeToCurrentGraph(node, app) {
    if (!node || !app) {
        return false;
    }
    const graphCandidates = [
        app?.graph,
        app?.canvas?.graph,
        app?.canvas?.ds?.graph,
    ];
    for (const graph of graphCandidates) {
        if (!graph) {
            continue;
        }
        // Try multiple signatures used by different frontend versions.
        // Prefer legacy `add` first: in some Node 2.0 builds `addNode` yields partial UI state.
        const attempts = [
            ["add", false],
            ["add", undefined],
            ["addNode", false],
            ["addNode", undefined],
        ];
        for (const [methodName, extraArg] of attempts) {
            const called = await callGraphAdd(graph, methodName, node, extraArg);
            if (!called) {
                continue;
            }
            if (isNodeInGraph(node, graph)) {
                normalizeNodeSizeAfterInsert(node);
                return true;
            }
        }
    }
    return false;
}

/**
 * Mark canvas/graph as dirty so UI refreshes after insertion.
 */
export function markNodeCanvasDirty(app) {
    if (typeof app?.graph?.setDirtyCanvas === "function") {
        app.graph.setDirtyCanvas(true, true);
        return;
    }
    if (typeof app?.canvas?.setDirty === "function") {
        app.canvas.setDirty(true, true);
    }
}

/**
 * Bring node into viewport when canvas API provides explicit centering.
 */
export function focusNodeInCanvas(node, app) {
    if (typeof app?.canvas?.centerOnNode === "function") {
        app.canvas.centerOnNode(node);
    }
}

/**
 * Create LiteGraph node from catalog metadata using internal/display name fallback.
 */
export async function createNodeFromCatalogInfo(nodeInfo, liteGraphObj = LiteGraph, options = {}) {
    const candidates = [nodeInfo?.node_name, nodeInfo?.display_name].filter(Boolean);
    const pos = Array.isArray(options?.pos) ? options.pos : getCanvasCenterInsertPos(options?.app);
    for (const name of candidates) {
        const createAttempts = [
            () => liteGraphObj?.createNode?.(name, nodeInfo?.display_name, { pos }),
            () => liteGraphObj?.createNode?.(name, null, { pos }),
            () => liteGraphObj?.createNode?.(name, undefined, { pos }),
            () => liteGraphObj?.createNode?.(name),
        ];
        for (const createAttempt of createAttempts) {
            const created = createAttempt();
            const node = created && typeof created.then === "function"
                ? await created
                : created;
            if (!node) {
                continue;
            }
            if (!Array.isArray(node.pos) || node.pos.length < 2) {
                node.pos = [safeNumber(pos[0], 200), safeNumber(pos[1], 120)];
            }
            return node;
        }
    }
    return null;
}
