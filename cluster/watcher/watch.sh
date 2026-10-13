#!/usr/bin/env bash
# Training watcher (RULES.md). Runs health.py every 10 min and an eval of every watched run's latest
# checkpoint every EVAL_PERIOD_MIN; exits, which wakes Claude, when health.py raises a flag, when an
# eval finishes or fails, or after MAX_MIN as a heartbeat. Start it again after each wake.
set -uo pipefail
W=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/watcher
ROOT=/scratch/sselvaraj/projects/gait-net-train/gaitnet
PY=/scratch/sselvaraj/isaac-gaitnet/venv/bin/python
# void fractions of every eval; 0.5 added 2026-10-13 because SC-idm saturates 0.1-0.3 with the IDM
DIFFS="0.0 0.1 0.2 0.3 0.4 0.5"
EVAL_PERIOD_MIN=${EVAL_PERIOD_MIN:-120}
MAX_MIN=${MAX_MIN:-180}
started=$(date +%s)

submit_eval() {
  local args="" labels="" name d m
  while read -r job name; do
    [[ -z "$job" || "$job" == \#* ]] && continue
    d=$(ls -d "$ROOT"/training/gaitnet/runs/gaitnet_2026*_"$name" 2>/dev/null | sort | tail -1)
    m=$(ls "$d"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$m" ] && { args="$args $m"; labels="$labels $name"; }
  done < "$W/runs.txt"
  local out="$ROOT/training/gaitnet/evals/$(date +%Y%m%d_%H%M)"
  local jid
  jid=$(cd "$ROOT" && sbatch --parsable --job-name=gaitnet-eval cluster/eval.sbatch --checkpoints $args --labels $labels --inference argmax marginal --difficulties $DIFFS --out "$out")
  # the same checkpoints with the literal IDM of the training controller, as they would be deployed
  local idm="$ROOT/training/gaitnet/idm_models/gaitnet_inverse.pt"
  [ -f "$idm" ] && (cd "$ROOT" && sbatch --parsable --job-name=gaitnet-eval-idm cluster/eval.sbatch --checkpoints $args --labels $labels --idm_inverse "$idm" --difficulties $DIFFS --out "${out}-idm") >> "$W/health.log"
  # and on an OCS2-like executor in Isaac (no IDM): the controller-agnostic test (decisions.md 2026-10-12 17:15)
  (cd "$ROOT" && sbatch --parsable --job-name=gaitnet-eval-emu cluster/eval.sbatch --checkpoints $args --labels $labels \
     --ctrl travel_compensation=1 touchdown_along=-0.04 touchdown_outboard=0.04 touchdown_noise=0.05 liftoff_delay_ticks=1 \
     --difficulties $DIFFS --out "${out}-emu") >> "$W/health.log"
  echo "$jid $out $(date +%s)" > "$W/current_eval"
  date +%s > "$W/last_eval_time"
  echo "$(date -u +%FT%TZ) submitted eval $jid -> $out" >> "$W/health.log"
}

while true; do
  out=$($PY "$W/health.py" 2>&1)
  { echo "== $(date -u +%FT%TZ)"; echo "$out"; } >> "$W/health.log"
  # a flag listed in acked.txt (one "RULE:run" prefix per line, decisions.md says why) no longer wakes
  flags=$(echo "$out" | grep '^FLAGS:' | sed 's/^FLAGS: *//; s/^none$//' | tr ' ' '\n' | grep -v '^$')
  [ -s "$W/acked.txt" ] && flags=$(echo "$flags" | grep -v -F -f "$W/acked.txt")
  if [ -n "$flags" ]; then echo "WAKE: FLAGS: $(echo $flags)"; echo "$out"; exit 0; fi

  if [ -f "$W/current_eval" ]; then
    read -r jid eout esub < "$W/current_eval"
    elog="$ROOT/cluster/logs/gaitnet-eval-$jid.out"
    state=$(squeue -h -j "$jid" -o %T 2>/dev/null)
    if [ -f "$elog" ] && grep -q "GAITNET-EVAL-DONE" "$elog"; then
      rm -f "$W/current_eval"
      echo "WAKE: eval $jid done -> $eout"; echo "$out"; $PY "$W/eval_report.py" --detail; exit 0
    elif [ -z "$state" ]; then
      rm -f "$W/current_eval"
      echo "WAKE: eval $jid ended without EVAL-DONE (log $elog)"; tail -30 "$elog" 2>/dev/null; exit 0
    elif [ $(( ($(date +%s) - esub) / 60 )) -ge 150 ]; then
      echo "WAKE: eval $jid still $state after 150 min"; exit 0
    fi
  else
    last=$(cat "$W/last_eval_time" 2>/dev/null || echo 0)
    if [ $(( ($(date +%s) - last) / 60 )) -ge "$EVAL_PERIOD_MIN" ]; then submit_eval; fi
  fi

  if [ $(( ($(date +%s) - started) / 60 )) -ge "$MAX_MIN" ]; then
    echo "WAKE: heartbeat after $MAX_MIN min"; echo "$out"; $PY "$W/eval_report.py"; exit 0
  fi
  sleep 600
done
