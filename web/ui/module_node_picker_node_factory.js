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
    const area = app?.canvas?.visible_area;
    if (area && area.length >= 4) {
        node.pos = [
            area[0] + area[2] * 0.5 - node.size[0] * 0.5,
            area[1] + area[3] * 0.5 - node.size[1] * 0.5,
        ];
        return;
    }
    node.pos = [200, 120];
}

/**
 * Create LiteGraph node from catalog metadata using internal/display name fallback.
 */
export function createNodeFromCatalogInfo(nodeInfo, liteGraphObj = LiteGraph) {
    const candidates = [nodeInfo?.node_name, nodeInfo?.display_name].filter(Boolean);
    for (const name of candidates) {
        const node = liteGraphObj?.createNode?.(name);
        if (node) {
            return node;
        }
    }
    return null;
}
