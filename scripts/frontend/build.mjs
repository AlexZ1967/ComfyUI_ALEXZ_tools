import * as esbuild from "esbuild";

await esbuild.build({
  entryPoints: [
    "frontend/src/hello.ts",
  ],

  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: true,

  // Эти импорты предоставляет сам ComfyUI во время работы браузера.
  // esbuild не должен пытаться искать и включать их в bundle.
  external: [
    "../../../scripts/app.js",
    "../../../scripts/api.js",
  ],

  outfile: "web/generated/hello.js",
});