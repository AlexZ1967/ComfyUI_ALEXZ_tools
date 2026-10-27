import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

const source = await readFile(new URL("../../frontend/src/workflow_file_save.ts", import.meta.url), "utf8");
const compiled = await transform(source, { loader: "ts", format: "esm", target: "es2022" });
const { WorkflowFileSaver } = await import(`data:text/javascript;base64,${Buffer.from(compiled.code).toString("base64")}`);

function handle(name) {
    const state = { text: "", mtime: 1, writes: 0, aborts: 0, fail: false };
    return {
        name, state,
        async getFile() { return { lastModified: state.mtime, size: Buffer.byteLength(state.text) }; },
        async createWritable() {
            state.writes += 1;
            let pending;
            return {
                async write(text) { if (state.fail) throw new Error("Disk failure"); pending = text; },
                async close() { state.text = pending; state.mtime += 1; },
                async abort() { state.aborts += 1; },
            };
        },
    };
}
let graph = "first";
let value = 1;
const files = [handle("Фильм.json"), handle("Second.json"), handle("Copy.json")];
const calls = [];
const order = [];
const saver = new WorkflowFileSaver({
    graphId: () => graph,
    workflow: async () => { order.push("serialize"); return { nodes: [{ value }], links: [] }; },
    suggestedName: () => "workflow.json",
    picker: () => async options => { order.push("picker"); calls.push(options); return files[calls.length - 1]; },
});
assert.deepEqual(await saver.save(), { cancelled: false, filename: "Фильм.json" });
assert.deepEqual(order.slice(0, 2), ["picker", "serialize"]);
assert.equal(calls[0].id, "alexz-workflow-json");
assert.deepEqual(calls[0].types[0].accept, { "application/json": [".json"] });
assert.equal(JSON.parse(files[0].state.text).nodes[0].value, 1);
value = 2;
await saver.save();
assert.equal(calls.length, 1);
assert.equal(JSON.parse(files[0].state.text).nodes[0].value, 2);
graph = "second";
await saver.save();
assert.equal(calls.length, 2);
graph = "first";
await saver.save();
assert.equal(calls.length, 2);
files[0].state.text = "external";
files[0].state.mtime += 1;
await assert.rejects(saver.save(), /изменён другим приложением/);
assert.equal(files[0].state.text, "external");
await saver.save(true);
assert.equal(calls[2].startIn, files[0]);
assert.equal(calls[2].suggestedName, "Фильм.json");
assert.equal(JSON.parse(files[2].state.text).nodes[0].value, 2);
value = 3;
await saver.save();
assert.equal(calls.length, 3);
assert.equal(JSON.parse(files[2].state.text).nodes[0].value, 3);
assert.equal(files[0].state.text, "external");

let serializeCount = 0;
const cancelled = new WorkflowFileSaver({
    graphId: () => "cancelled", suggestedName: () => "x.json",
    workflow: async () => { serializeCount += 1; return { nodes: [], links: [] }; },
    picker: () => async () => { throw Object.assign(new Error("Cancelled"), { name: "AbortError" }); },
});
assert.deepEqual(await cancelled.save(true), { cancelled: true });
assert.equal(serializeCount, 0);
const unsupported = new WorkflowFileSaver({
    graphId: () => "unsupported", suggestedName: () => "x.json", picker: () => undefined,
    workflow: async () => { throw new Error("Must not serialize"); },
});
await assert.rejects(unsupported.save(), /не поддерживает системный диалог/);

const brokenFile = handle("broken.json");
brokenFile.state.text = "original";
brokenFile.state.fail = true;
let pickerCalls = 0;
const broken = new WorkflowFileSaver({
    graphId: () => "broken", suggestedName: () => "x.json",
    workflow: async () => ({ nodes: [], links: [] }),
    picker: () => async () => { pickerCalls += 1; return brokenFile; },
});
await assert.rejects(broken.save(), /Disk failure/);
assert.equal(brokenFile.state.aborts, 1);
assert.equal(brokenFile.state.text, "original");
brokenFile.state.fail = false;
await broken.save();
assert.equal(pickerCalls, 2);

let active = "a";
const changed = new WorkflowFileSaver({
    graphId: () => active, suggestedName: () => "x.json",
    workflow: async () => ({ nodes: [], links: [] }),
    picker: () => async () => { active = "b"; return brokenFile; },
});
const writes = brokenFile.state.writes;
await assert.rejects(changed.save(), /Активный workflow изменился/);
assert.equal(brokenFile.state.writes, writes);

let resolvePicker;
const pending = new WorkflowFileSaver({
    graphId: () => "pending", suggestedName: () => "x.json",
    workflow: async () => ({ nodes: [], links: [] }),
    picker: () => () => new Promise(resolve => { resolvePicker = resolve; }),
});
const first = pending.save();
await assert.rejects(pending.save(), /уже выполняется/);
resolvePicker(handle("pending.json"));
await first;
console.log("Native workflow save behavior: PASS");
