#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Dev-запуск ComfyUI для ALEXZ_tools и Silent Film.

Использование:
  bash scripts/start_comfyui_dev.sh [--dry-run] [--extra-node ИМЯ] [-- параметры ComfyUI]

По умолчанию загружаются только ALEXZ_tools, ComfyUI-VideoHelperSuite и comfyui-manager.
Conda environment: p313-torch214-cu132. Порт ComfyUI по умолчанию: 8188.

Примеры:
  bash scripts/start_comfyui_dev.sh --dry-run
  bash scripts/start_comfyui_dev.sh -- --port 8189
  bash scripts/start_comfyui_dev.sh --extra-node was-node-suite-comfyui

Параметры:
  --dry-run          Показать команду без запуска Python и Conda.
  --extra-node ИМЯ   Дополнительная папка/файл из custom_nodes; можно повторять.
  -h, --help         Показать справку.

Окружение:
  ALEXZ_COMFY_ROOT   Путь к ComfyUI, если проект находится вне custom_nodes.
  CONDA_EXE         Путь к исполняемому файлу Conda; иначе используется PATH.
EOF
}

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
comfy_root="${ALEXZ_COMFY_ROOT:-$(cd -- "$project_root/../.." && pwd)}"
project_name="$(basename -- "$project_root")"
allowed_nodes=("$project_name" "ComfyUI-VideoHelperSuite" "comfyui-manager")
comfy_args=()
dry_run=false

while (($#)); do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --extra-node)
      if (($# < 2)) || [[ -z "$2" || "$2" == -* || "$2" == */* ]]; then
        echo 'Ошибка: --extra-node требует имя папки или файла из custom_nodes.' >&2
        exit 2
      fi
      allowed_nodes+=("$2")
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    --) shift; comfy_args+=("$@"); break ;;
    *) comfy_args+=("$1"); shift ;;
  esac
done

if [[ ! -f "$comfy_root/main.py" ]]; then
  echo "Ошибка: main.py не найден в $comfy_root. Задайте ALEXZ_COMFY_ROOT." >&2
  exit 2
fi
for node_name in "${allowed_nodes[@]}"; do
  if [[ ! -e "$comfy_root/custom_nodes/$node_name" ]]; then
    echo "Ошибка: отсутствует $comfy_root/custom_nodes/$node_name." >&2
    exit 2
  fi
done

conda_bin="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_bin" ]]; then
  echo 'Ошибка: Conda не найдена. Добавьте conda в PATH или задайте CONDA_EXE.' >&2
  exit 2
fi

# Флаги ограничивают и импорт пакетов, и их prestartup scripts в установленном ComfyUI.
command_args=("$conda_bin" run --no-capture-output -n p313-torch214-cu132 python -u "$comfy_root/main.py"
  --disable-all-custom-nodes --disable-partner-nodes --whitelist-custom-nodes "${allowed_nodes[@]}" "${comfy_args[@]}")

printf 'Dev ComfyUI: %s\n' "$comfy_root"
printf 'Разрешены custom nodes: %s\n' "${allowed_nodes[*]}"
if "$dry_run"; then
  printf 'Команда: '
  printf '%q ' "${command_args[@]}"
  printf '\n'
  exit 0
fi

# Тот же ComfyUI и каталоги данных; меняется только набор загружаемых расширений.
cd -- "$comfy_root"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
exec "${command_args[@]}"
