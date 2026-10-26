import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const extensions = [];
const nodes = new Map();
const graph = { getNodeById(id) { return nodes.get(id); }, _nodes: [] };
const app = { graph, registerExtension(extension) { extensions.push(extension); } };
const source = await readFile(new URL("../../web/silent_film_preview.js", import.meta.url), "utf8");
new Function("app", source.replace(/^import .*?;$/gm, ""))(app);
const extension = extensions[0];
let previousHooks = 0;
class Preview {
    constructor() { this.type = this.comfyClass = "PreviewImage"; this.graph = graph; this.link = null; }
    getInputLink() { return this.link; }
    onNodeCreated() { previousHooks++; }
    onConfigure() { previousHooks++; }
    onConnectionsChange() { previousHooks++; }
}
extension.beforeRegisterNodeDef(Preview, { name: "PreviewImage" });
const preview = new Preview(); graph._nodes.push(preview);
nodes.set(2, { id: 2, type: "VideoSilentFilmCamera", outputs: [{ name: "image" }, { name: "film_json" }, { name: "preview" }] });
preview.link = { origin_id: 2, origin_slot: 2 };
preview.onNodeCreated();
assert.equal(preview.comfyClass, "PreviewImage", "Do not use a backend absent from old running servers");
extension.beforeRegisterNodeDef(class {}, { name: "VideoPreviewFrame" });
const detached = new Preview(); detached.graph = null;
detached.getInputLink = () => { throw Error("Node has not been added to a graph yet"); };
detached.onNodeCreated();
assert.equal(detached.comfyClass, "PreviewImage");
extension.afterConfigureGraph();
assert.equal(preview.comfyClass, "VideoPreviewFrame");
assert.equal(preview.type, "PreviewImage", "Saved workflow node type must stay unchanged");
for (const [type, slot, outputName = "preview"] of [["VideoSilentFilmCamera", 2], ["VideoSilentFilmNegative", 2], ["VideoSilentFilmPrint", 2], ["VideoSilentFilmProjector", 3], ["VideoSilentFilmNegative", 3, "preview_positive"]]) {
    const outputs = Array.from({ length: slot + 1 }, (_, index) => ({ name: index === slot ? outputName : "image" }));
    nodes.set(2, { id: 2, type, mode: 4, outputs });
    preview.link.origin_slot = slot;
    preview.onConnectionsChange();
    assert.equal(preview.comfyClass, "VideoPreviewFrame", type);
}
preview.link.origin_slot = 0;
preview.onConnectionsChange();
assert.equal(preview.comfyClass, "PreviewImage", "A regular full-batch preview link keeps its contract");
nodes.set(9, { id: 9, type: "Reroute", getInputLink: () => ({ origin_id: 2, origin_slot: 3 }) });
preview.link = { origin_id: 9, origin_slot: 0 };
preview.onConfigure();
assert.equal(preview.comfyClass, "VideoPreviewFrame");
preview.comfyClass = "OtherExtensionPreview";
preview.onConnectionsChange();
assert.equal(preview.comfyClass, "OtherExtensionPreview");
preview.comfyClass = "PreviewImage";
preview.onConnectionsChange();
preview.link = null; preview.onConnectionsChange();
assert.equal(preview.comfyClass, "PreviewImage");
assert(previousHooks > 5);
console.log("Silent Film single-frame preview routing, bypass and restoration: PASS");
