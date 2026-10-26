import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { transform } from "esbuild";

// Проверяем перенос на настоящем контракте Python, без копии схемы в JS-тесте.
const allSchemas = JSON.parse(execFileSync("python", ["-c", `
import importlib,json,sys,types
from pathlib import Path
for name,path in [("ComfyUI_ALEXZ_tools",Path.cwd()),("ComfyUI_ALEXZ_tools.nodes",Path.cwd()/"nodes")]:
    package=types.ModuleType(name);package.__path__=[str(path)];sys.modules[name]=package
m=importlib.import_module("ComfyUI_ALEXZ_tools.nodes.video_silent_film_pipeline")
result=[]
for stage in ["Negative","Print","Projector"]:
    name="VideoSilentFilm"+stage;c=getattr(m,name);i=c.INPUT_TYPES()
    result.append({"name":name,"input":i,"input_order":{k:list(v) for k,v in i.items()}})
print(json.dumps(result))
`], { encoding: "utf8", cwd: new URL("../..", import.meta.url) }));
const schemas = allSchemas.slice(0, 2);

const extensions = [];
const source = readFileSync(new URL("../../frontend/src/silent_film_emulsion.ts", import.meta.url), "utf8");
const compiled = await transform(source, { loader: "ts", target: "es2022", format: "esm" });
new Function("app", compiled.code.replace(/^import .*?;$/gm, ""))({ registerExtension(extension) { extensions.push(extension); } });
const extension = extensions[0];
const legacy = {
    VideoSilentFilmNegative: ["exposure_ev", "development_contrast", "base_fog", "density_range", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
    VideoSilentFilmPrint: ["print_exposure_ev", "print_contrast", "reference_density", "base_fog", "density_range", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
};
const oldValues = {
    exposure_ev: 0.25, development_contrast: 0.9, base_fog: 0.1, density_range: 2.2, development_variation: 0.01,
    print_exposure_ev: -0.2, print_contrast: 1.1, reference_density: 1.2, print_softness: 0, tone_mode: "neutral_bw",
    grain_strength: 0.02, grain_size: 3, seed: 1925, control_after_generate: "fixed", compute_device: "cpu", batch_size: 2,
};
let originalCalls = 0;
class Node {
    onSerialize() { originalCalls++; }
}
const originalSerialize = Node.prototype.onSerialize;
extension.beforeRegisterNodeDef(Node, { name: "VideoSilentFilmNegative", input: { required: { density_range: ["FLOAT", {}] } } });
assert.equal(Node.prototype.onSerialize, originalSerialize, "Do not change nodes registered by an old backend");
for (const schema of allSchemas) extension.beforeRegisterNodeDef(Node, schema);

for (const named of [true, false]) for (const objectLinks of [true, false]) {
    const nodes = schemas.map((schema, i) => {
        const names = ["image", "film_json", ...legacy[schema.name].filter((name) => name !== "control_after_generate")];
        return {
            id: i + 1, type: schema.name,
            inputs: names.map((name) => ({ name, type: name === "image" ? "IMAGE" : "FLOAT", link: name === "seed" ? i + 10 : null })),
            outputs: [{ name: "negative", type: "IMAGE", links: null }, { name: "film_json", type: "STRING" }, { name: "preview", type: "IMAGE" }],
            widgets_values: legacy[schema.name].map((name) => oldValues[name]),
            ...(named ? { widgets_values_named: { ...oldValues, seed: 888 } } : {}),
        };
    });
    const links = nodes.map((node, i) => [i + 10, 99, 0, node.id, node.inputs.findIndex((input) => input.name === "seed"), "INT"]);
    const graph = { nodes, links: objectLinks ? links.map(([id, origin_id, origin_slot, target_id, target_slot, type]) => ({ id, origin_id, origin_slot, target_id, target_slot, type })) : links };
    extension.beforeConfigureGraph(graph);
    for (const [i, node] of nodes.entries()) {
        const values = node.widgets_values_named;
        assert.equal(values.seed, named ? 888 : 1925);
        assert.equal(values.fog_strength, 0);
        assert.equal(values.shadow_softness, 0);
        assert.equal(values.highlight_softness, 0);
        assert.equal(values.grain_strength, 0.02);
        assert.equal(values.compute_device, "cpu");
        assert.equal(values.control_after_generate, "fixed");
        assert(!node.inputs.some((input) => ["density_range", "reference_density", "base_fog"].includes(input.name)));
        const link = graph.links[i];
        const targetSlot = objectLinks ? link.target_slot : link[4];
        assert.equal(node.inputs[targetSlot].name, "seed");
        assert.equal(node.inputs[targetSlot].link, i + 10);
        assert.equal(node.properties.alexz_emulsion_model, "alexz.neutral_bw_halation_v2");
        assert.equal(values.halation_strength, 0);
        assert.equal(values.halation_size, 0.15);
    }
    assert.equal(nodes[0].outputs[3].name, "preview_positive");
    assert.equal(nodes[0].widgets_values_named.input_transfer, "srgb");
    assert.equal(nodes[1].widgets_values_named.output_transfer, "from_negative");
    assert.equal(nodes[0].widgets_values_named.exposure_ev, 0.25);
    assert.equal(nodes[1].widgets_values_named.print_exposure_ev, -0.2);
    const migrated = JSON.stringify(graph);
    extension.beforeConfigureGraph(graph);
    assert.equal(JSON.stringify(graph), migrated, "Migration must be idempotent");
}

const sourceNode = { id: 99, outputs: [{ links: [51, 52] }] };
const targetNode = {
    id: 1, type: "VideoSilentFilmNegative",
    inputs: [{ name: "image", type: "IMAGE", link: null }, { name: "base_fog", type: "FLOAT", link: 51 }, { name: "seed", type: "INT", link: 52 }],
    widgets_values: legacy.VideoSilentFilmNegative.map((name) => oldValues[name]),
};
const disconnected = { nodes: [sourceNode, targetNode], links: [[51, 99, 0, 1, 1, "FLOAT"], [52, 99, 0, 1, 2, "INT"]] };
const warnings = [];
const warn = console.warn;
try {
    console.warn = (message) => warnings.push(message);
    extension.beforeConfigureGraph(disconnected);
} finally {
    console.warn = warn;
}
assert.equal(warnings.length, 1);
assert.deepEqual(sourceNode.outputs[0].links, [52]);
assert.equal(disconnected.links.length, 1);
assert.equal(targetNode.inputs[disconnected.links[0][4]].name, "seed");

const preHalationFields = [
    ["input_transfer", "exposure_ev", "development_contrast", "shadow_softness", "highlight_softness", "fog_strength", "development_variation", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
    ["output_transfer", "print_exposure_ev", "print_contrast", "shadow_softness", "highlight_softness", "fog_strength", "print_softness", "tone_mode", "grain_strength", "grain_size", "seed", "control_after_generate", "compute_device", "batch_size"],
];
for (const named of [true, false]) for (const [index, schema] of schemas.entries()) {
    const names = preHalationFields[index];
    const saved = { ...oldValues, input_transfer: "linear", output_transfer: "from_negative", shadow_softness: 0.2, highlight_softness: 0.3, fog_strength: 0.1 };
    const node = { id: 23, type: schema.name, properties: { alexz_emulsion_model: "alexz.neutral_bw_v1" }, inputs: [{ name: "image", link: null }, { name: "film_json", link: null }, ...names.filter(n => n !== "control_after_generate").map(name => ({ name, link: name === "seed" ? 29 : null }))], widgets_values: names.map(name => saved[name]), ...(named ? { widgets_values_named: saved } : {}) };
    const graph = { nodes: [node], links: [[29, 99, 0, 23, node.inputs.findIndex(i => i.name === "seed"), "INT"]] };
    extension.beforeConfigureGraph(graph);
    const values = node.widgets_values_named;
    for (const field of names) assert.equal(values[field], saved[field], `Keep pre-halation ${field}`);
    assert.equal(values.halation_strength, 0);
    assert.equal(values.halation_size, 0.15);
    assert.equal(node.inputs[graph.links[0][4]].name, "seed");
    const again = JSON.stringify(graph); extension.beforeConfigureGraph(graph);
    assert.equal(JSON.stringify(graph), again);
}

const projectionNames = ["projection_fps", "gate_weave_px", "softness", "vignette_strength", "edge_softness", "edge_start", "edge_end", "lamp_flicker", "lamp_breathing", "seed", "control_after_generate", "compute_device", "batch_size"];
for (const named of [true, false]) {
    const values = [25, 2, 0.02, 0.1, 0.15, 0.8, 1.3, 0.035, 0.015, 123, "fixed", "cpu", 10];
    const projector = { id: 8, type: "VideoSilentFilmProjector", inputs: [{ name: "image", type: "IMAGE", link: null }, { name: "film_json", type: "STRING", link: null }, ...projectionNames.filter(n => n !== "control_after_generate").map(name => ({ name, type: "FLOAT", link: name === "seed" ? 82 : null }))], widgets_values: values, ...(named ? { widgets_values_named: Object.fromEntries(projectionNames.map((n, i) => [n, values[i]])) } : {}) };
    const graph = { nodes: [projector], links: [[82, 99, 0, 8, projector.inputs.findIndex(i => i.name === "seed"), "INT"]] };
    extension.beforeConfigureGraph(graph);
    assert.equal(projector.widgets_values_named.glow_strength, 0);
    assert.equal(projector.widgets_values_named.glow_size, 1);
    assert.equal(projector.widgets_values_named.softness, 0.02);
    assert.equal(projector.widgets_values_named.lamp_flicker, 0.035);
    assert.equal(projector.widgets_values_named.seed, 123);
    assert.equal(projector.widgets_values_named.batch_size, 10);
    assert.equal(projector.inputs[graph.links[0][4]].name, "seed");
    assert.equal(projector.properties.alexz_projection_model, "alexz.projection_glow_v1");
    const migrated = JSON.stringify(graph);
    extension.beforeConfigureGraph(graph);
    assert.equal(JSON.stringify(graph), migrated);
}

const workflow = JSON.parse(readFileSync(new URL("../../workflows/Видео-в-Хронику-по-этапам.json", import.meta.url), "utf8"));
const original = JSON.stringify(workflow);
extension.beforeConfigureGraph(workflow);
assert.equal(JSON.stringify(workflow), original, "The updated example must not be treated as legacy");
const instance = new Node();
instance.widgets = [{ name: "exposure_ev", value: 0.5 }];
const serialized = {};
instance.onSerialize(serialized);
assert(originalCalls > 0);
assert.equal(serialized.widgets_values_named.exposure_ev, 0.5);
assert.equal(serialized.properties.alexz_emulsion_model, "alexz.neutral_bw_halation_v2");
console.log("Silent Film emulsion schema migration, seed links, named values and serialization: PASS");
