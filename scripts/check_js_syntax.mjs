/** Recursively validate JavaScript syntax with the active Node.js runtime. */

import { spawnSync } from "node:child_process";
import { readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const SUPPORTED_EXTENSIONS = new Set([".js", ".mjs"]);

async function collectJavaScriptFiles(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    const files = [];
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
        const entryPath = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            files.push(...await collectJavaScriptFiles(entryPath));
        } else if (entry.isFile() && SUPPORTED_EXTENSIONS.has(path.extname(entry.name))) {
            files.push(entryPath);
        }
    }
    return files;
}

const root = path.resolve(process.argv[2] || "web");
let files;
try {
    files = await collectJavaScriptFiles(root);
} catch (error) {
    console.error(`js-check-all: cannot scan ${root}: ${error.message}`);
    process.exit(1);
}

if (files.length === 0) {
    console.error(`js-check-all: no JavaScript files found under ${root}`);
    process.exit(1);
}

for (const file of files) {
    const result = spawnSync(process.execPath, ["--check", file], {
        encoding: "utf8",
    });
    if (result.status !== 0) {
        process.stderr.write(result.stdout || "");
        process.stderr.write(result.stderr || "");
        console.error(`js-check-all: failed: ${path.relative(process.cwd(), file)}`);
        process.exit(result.status || 1);
    }
}

console.log(`js-check-all: OK (${files.length} files under ${path.relative(process.cwd(), root) || "."})`);
