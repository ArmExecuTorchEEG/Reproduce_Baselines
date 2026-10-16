#!/usr/bin/env bash
# Run only after the user has activated Arm tools and personally accepted licenses.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: bash src/fvp/run.sh <prepared-project-directory> <new-console-log>' >&2
  exit 2
fi
project="$(cd "$1" && pwd)"
log="$2"
[[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
[[ "${FVP_EULA_ACCEPTED:-}" == '11.32.23' ]] || {
  echo 'Review/accept https://artifacts.tools.arm.com/avh/11.32.23/license_agreement.txt personally, then set FVP_EULA_ACCEPTED=11.32.23.' >&2
  exit 2
}
command -v cbuild >/dev/null || { echo 'Activate CMSIS-Toolbox/Arm tools first.' >&2; exit 2; }
command -v docker >/dev/null || { echo 'Docker CLI missing from PATH.' >&2; exit 2; }
docker info >/dev/null
# No --packs: install/approve pack licenses in Keil Studio beforehand.
(cd "$project" && cbuild setup cmsis-executorch.csolution.yml --active SSE-320-U85)
# Deliberately DO NOT invoke create_ai_layer.py: the unchanged EEG PTE is staged.
(cd "$project" && cbuild cmsis-executorch.csolution.yml --active SSE-320-U85)
(cd "$project" && bash .vscode/fvp.sh -f board/Corstone-320/fvp_config.txt --simlimit 60 \
  -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.hex) 2>&1 | tee "$log"
