#!/usr/bin/env bash
# After gen_synth_tts.py finishes: QC -> build voice_dataset_v2 -> Colab zip.
# Usage: scripts/run_synth_pipeline.sh [PID_OF_GENERATION_TO_WAIT_FOR]
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${1:-}" ]]; then
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
ML=sim_vision/python/.venv/bin/python
TTS=tools/tts/.venv/bin/python
# generation is resumable: re-run once to fill any job that failed or was interrupted
PYTORCH_ENABLE_MPS_FALLBACK=1 $TTS scripts/gen_synth_tts.py
$ML scripts/qc_synth_asr.py
$ML scripts/build_dataset_v2.py
cd data/processed
rm -f voice_dataset_v2_colab.zip
zip -r -q voice_dataset_v2_colab.zip voice_dataset_v2
python3 -c "import zipfile;z=zipfile.ZipFile('voice_dataset_v2_colab.zip');assert z.testzip() is None;print('zip ok',len(z.namelist()),'files')"
ls -la voice_dataset_v2_colab.zip
echo PIPELINE_DONE
