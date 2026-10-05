# AGENTS Instructions

## Language and Communication

- Communicate with the user in Russian by default.
- Describe plans, actions, findings, test results, refactoring steps, and explanations in Russian.
- Ask clarification questions in Russian.
- Summaries of completed work must be in Russian.
- Write code comments and project documentation in Russian unless a specific file, API, external standard, or existing convention requires English.
- Keep user-facing documentation, guides, changelogs, refactoring notes, and internal project notes in Russian by default.
- Write function and method docstrings/descriptions inside source code in English.
- Keep identifiers, function names, class names, variable names, type names, API names, protocol names, and technical symbols in English.
- Do not translate established programming terms, library names, ComfyUI symbols, API names, or command-line options.
- If an existing file already follows a different language convention, preserve its established style unless explicitly asked to change it.

## Runtime Environment

- Always run local checks and tests inside Conda env `p313-torch214-cu132`.
- Preferred command prefix: `conda run -n p313-torch214-cu132`.
- Do not report a test result as valid if it was run outside `p313-torch214-cu132`.
- If a command was run without `p313-torch214-cu132`, rerun it with `conda run -n p313-torch214-cu132` before reporting results.

## Canonical Commands

- Docs check: `conda run -n p313-torch214-cu132 python utils/docs_check.py`
- Seam smoke tests: `conda run -n p313-torch214-cu132 pytest -q tests/test_smoke_nodes.py -k seam_match`
- Full smoke tests: `conda run -n p313-torch214-cu132 pytest -q tests/test_smoke_nodes.py`
- All JS syntax checks: `make js-check-all`
- Module Node Picker behavior test: `make js-test`
- TypeScript check: `make ts-check`
- TypeScript build: `make ts-build`
- TypeScript runtime bundle: `make frontend-build`
- Full Python, JavaScript, TypeScript, and docs validation: `make test`

## Project Boundaries

- The editable project root is `/mnt/comfy/ComfyUI/custom_nodes/ComfyUI_ALEXZ_tools`.
- Normal implementation changes must remain inside `/mnt/comfy/ComfyUI/custom_nodes/ComfyUI_ALEXZ_tools`.
- The installed ComfyUI repository is available at `/mnt/comfy/ComfyUI`.
- Treat `/mnt/comfy/ComfyUI` as read-only reference material.
- Never modify ComfyUI core files unless the user explicitly requests a core modification.
- Files elsewhere under `/mnt/comfy/ComfyUI/custom_nodes/` may be inspected for reference when useful, but must not be modified unless explicitly requested.

## ComfyUI Reference Strategy

- Use the locally installed ComfyUI source as the primary reference for backend APIs, server routes, frontend APIs, extension hooks, and runtime behavior.
- Prefer the APIs and behavior of the locally installed ComfyUI version over assumptions, remembered APIs, or outdated internet examples.
- When implementing ComfyUI integrations, inspect the relevant local source before introducing a new API dependency.
- Do not modify ComfyUI core just to make ALEXZ_tools work unless the user explicitly requests that approach.

## Frontend Language Strategy

- Do not migrate existing JavaScript to TypeScript just for consistency.
- Keep small, simple, stable frontend modules in JavaScript when static typing adds little value.
- Prefer TypeScript for new complex frontend modules, especially when they involve:
  - shared application state;
  - backend API contracts;
  - non-trivial data structures;
  - multiple interacting modules;
  - reusable public interfaces;
  - code that is likely to be refactored frequently.
- When an existing JavaScript module becomes substantially more complex, consider migrating it to TypeScript as a separate, explicit refactor.
- Never mix a JS-to-TS migration into an unrelated bug fix unless required by the task.
- File size alone is not a sufficient reason to choose TypeScript; prefer TypeScript based on complexity, shared contracts, state, and refactoring risk.

## ComfyUI Frontend Boundaries

- `web/` contains frontend JavaScript that is served directly by ComfyUI through `WEB_DIRECTORY = "./web"`.
- Existing JavaScript under `web/` is runtime source and may be edited directly when appropriate.
- `frontend/src/` contains handwritten TypeScript source files.
- `frontend/dist/` contains generated JavaScript output from the TypeScript compiler.
- Do not manually edit files in `frontend/dist/`.
- Do not copy generated TypeScript output into `web/` unless the task explicitly requires deploying that module to ComfyUI.
- Keep generated artifacts separate from handwritten source code.
- Do not introduce a frontend build step into an unrelated bug fix unless necessary.

## Change Discipline

- Prefer minimal, focused changes.
- Do not perform unrelated cleanup, renaming, formatting, or refactoring unless explicitly requested.
- Preserve existing behavior unless the task requires changing it.
- Before changing a public API, node identifier, serialized workflow field, frontend contract, or backend route, inspect current usages first.
- Avoid broad rewrites when a local fix is sufficient.
- When a task touches both Python and frontend code, verify both sides of the contract.

## Validation

- Run the smallest relevant validation first.
- For frontend JavaScript changes, run `make js-check-all` and relevant JS behavior tests.
- For TypeScript changes, run `make ts-check`; run `make ts-build` when generated output or build validity matters.
- For Python changes, run the relevant targeted pytest tests before the full test suite.
- Before reporting substantial work as complete, prefer `make test` when practical.
- Report failing tests accurately and do not hide unrelated pre-existing failures.

## Generated and Local Files

- Do not commit or manually edit generated files unless the project explicitly requires them.
- `node_modules/` is local dependency state and must not be treated as source.
- `frontend/dist/` is generated output.
- Local VS Code configuration and machine-specific paths should not be treated as portable project source unless explicitly requested.

## Working Style

- Before making non-trivial changes, briefly state the intended plan in Russian.
- Keep the plan concise and focused on the requested task.
- Do not ask for confirmation when the task is clear and safe to execute.
- Prefer inspecting relevant existing code before editing.
- When modifying code, explain important decisions in Russian as you work.
- Avoid narrating trivial low-level operations.
- If a likely bug, incompatibility, or architectural risk is discovered, report it early before continuing.

## Completion Report

- After completing a task, provide a concise summary in Russian.
- Include:
  - what was changed;
  - which files were modified;
  - which tests or checks were run;
  - whether they passed;
  - any remaining risks, limitations, or follow-up work.
- Do not claim success unless the relevant checks actually passed.
- If some checks were not run, state that explicitly.

## Working Style

- Before making non-trivial changes, briefly state the intended plan in Russian.
- Keep the plan concise and focused on the requested task.
- Do not ask for confirmation when the task is clear and safe to execute.
- Prefer inspecting relevant existing code before editing.
- When modifying code, explain important decisions in Russian as you work.
- Avoid narrating trivial low-level operations.
- If a likely bug, incompatibility, or architectural risk is discovered, report it early before continuing.

## Completion Report

- After completing a task, provide a concise summary in Russian.
- Include:
  - what was changed;
  - which files were modified;
  - which tests or checks were run;
  - whether they passed;
  - any remaining risks, limitations, or follow-up work.
- Do not claim success unless the relevant checks actually passed.
- If some checks were not run, state that explicitly.

## Vibe Coding Guardrails

- Treat the user's request as the source of truth for scope and intent.
- Prefer implementing the smallest working change that satisfies the request.
- Do not broaden the task into unrelated refactoring or architecture cleanup.
- When multiple implementations are possible, prefer the one that is easiest to understand, test, and maintain.
- Preserve backward compatibility with existing workflows whenever practical.
- For ComfyUI nodes, avoid changing node names, input names, output names, categories, or serialized workflow behavior unless explicitly required.
- For frontend changes, preserve current ComfyUI behavior and interaction patterns unless the user requests a redesign.

## TypeScript Runtime Build

- TypeScript source lives under `frontend/src/`.
- `npm run ts-check` performs type checking only.
- Runtime TypeScript modules are bundled with esbuild using `npm run frontend-build`.
- Bundled runtime output is written to `web/generated/`.
- Files under `web/generated/` are generated and must not be edited manually.
- When a task changes runtime TypeScript code, run both `make ts-check` and `make frontend-build`.
- Do not place handwritten source files in `web/generated/`.
- Commit runtime bundles and their source maps under `web/generated/` so custom node installations work without a local npm build; keep `frontend/dist/` and `node_modules/` untracked.
- `make test` must validate TypeScript without generating runtime output.
