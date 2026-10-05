# Phase 7: Quality Audit

Дата аудита: 2026-07-23

## Exception boundaries

Широкий `except Exception` удален из deterministic JSON/date/path/snapshot,
Manager-file и component-registry helpers там, где ожидаемые ошибки имеют
известные типы.

Широкий перехват намеренно сохранен только на границах, где вызывается внешний
или runtime-код:

- ComfyUI API routes и background workers, которые обязаны вернуть
  диагностический status вместо падения UI;
- optional dependency imports и node execution adapters;
- callbacks от ComfyUI, Manager или сторонних custom nodes;
- fallback-аннотация произвольных node metadata/iterators;
- сетевые transport orchestration paths с несколькими optional backends.

`tests/test_exception_boundaries.py` запрещает возврат широких перехватов в
очищенные deterministic helpers.

## Compatibility shims

В `utils/module_browser/*.py` остаются 27 top-level compatibility wrappers.
Production-код использует canonical подпакеты и не импортирует эти wrappers,
что проверяется `tests/test_module_browser_shim_boundaries.py`.

Решение Phase 7: wrappers не удалять в patch/minor cleanup-релизе. Они являются
внешним backward-compatible import surface для пользовательских скриптов и
тестовых интеграций. Удаление допустимо только в следующем major migration
cycle с release note и проверкой внешних consumers.

## Documentation consistency

`utils/docs_check.py` теперь проверяет:

- совпадение версии в `pyproject.toml`, `README.md` и верхней записи
  `CHANGELOG.md`;
- существование локальных Markdown-ссылок в README и guides;
- наличие central `NodeSpec` metadata без orphan/missing записей;
- docstrings у документированных node classes и их `FUNCTION` methods;
- упоминание required и optional inputs, outputs и guide links.

## TODO / legacy comments

TODO/FIXME/HACK-маркеры в `propainter` и `utils` проверены. Два остатка в
`propainter/utils/image_utils.py` закрыты:

- `ImageOutpaintConfig` использует базовый `ImageConfig.__post_init__`;
- upstream tensor converter явно помечен как compatibility helper, а не
  неопределенная будущая задача.

Legacy-комментарии Module Node Picker оставлены только там, где описывают
действующий storage/state/import compatibility contract.
