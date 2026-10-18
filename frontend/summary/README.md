# Session summary (right panel · End Run / New Run)

Shown in the **right (cursor) panel** of the continuous streaming mode (`?stream=1`), in place of
the prediction view. Built only from `controller.records`: completed, validated hardware results.
The animated GUI counter ("Visualized accuracy") is never used. The legacy four-trial replay
(`?v=eeg-1`) is unchanged and is not summarized; its controls stay hidden there.

Views of the right panel (`#sessionSummary` sits inside `.cursor-panel`, absolutely positioned
over the prediction elements, which stay mounted so cursor/playback state is never lost):

| State | Right panel | Label |
|---|---|---|
| Playing | prediction / cursor view | — |
| Paused after the first Play | live summary rebuilt from current records; nothing finalized | Run paused |
| Controller stopped by an error or disconnect | live summary | Run interrupted |
| End Run finalized (frozen snapshot, kept in `history`) | final summary with New Run | Run complete, or Run interrupted when a request was unresolved |
| New Run / Reset | prediction view, waits for Play | — |

A result that completes after Pause updates the paused summary once; its visualization stays queued
for Resume. Before the first Play of a run the prediction view is shown even though the controller is
paused. No popup, backdrop, focus trap or close control; the panel fades in (disabled under
`prefers-reduced-motion`).

One centered primary action sits at the bottom of the panel: **Resume** while paused (only when the
controller is ready and has not failed), **New Run** on a finalized snapshot, nothing on an interrupted
live view. The panel's Resume clicks the top Play button, so both share the bridge's single Play
handler; while paused after a run started the top button reads "▶ Resume". Supporting lines use
" | " separators ("30 correct | 10 incorrect", "30 unique recordings | 10 repeat evaluations",
"Prerecorded EEG | 30 recordings | 3 artifact-flagged"); the EEG line and the amplitude note sit directly
under the Top EEG Channels ranking and wrap within that column.

## Files

| File | Role |
|---|---|
| `session-summary.js` | Pure metrics and the streaming-record adapter (`EdgeSessionSummary`). Node-testable, no DOM. |
| `eeg-overview.js` | EEG Signal Overview (`EdgeEEGOverview`): pooled channel RMS over unique completed recordings, separate from classification. |
| `model-metadata.js` | Verified model/experiment/dataset/class metadata keyed by PTE SHA-256 (`EdgeModelMetadata`). |
| `summary-view.js` | HTML rendering (`EdgeSummaryView.renderHTML`) and the `#sessionSummary` overlay binding. |
| `session-run.js` | End Run / New Run manager (`EdgeSessionRun`): wraps one existing controller, no second loop. |
| `summary.css` | Styles scoped to `#sessionSummary` and the two run buttons only. |
| `preview.html`, `preview-data.js` | Board-free review: the real GUI (mock mode) in a frame with its right panel switched between the prediction view and **synthetic** paused/complete summaries. Never loaded by `index.html`. |

Shared-file integration points (the complete list):

1. `index.html`: `summary.css` link; `endRunBtn` / `newRunBtn` (hidden by default) in `.controls`; the
   `#sessionSummary` container as the last child of `.cursor-panel`; five `summary/*.js` script tags
   before `streaming/bridge.js`.
2. `streaming/bridge.js`: the Reset handler body became `resetRun()` (shared with New Run), and one call after
   the verified handshake: `window.EdgeSessionRun?.attach({controller,dataset,firmware:controller.firmware,cancelVisualization:cancel,reset:resetRun})`.
3. `tests/design.cjs`: Cindy's stylesheet is verified as the byte-identical leading block, with a documented
   appended sizing block allowed. `tests/browser-cdp.mjs`: optional readiness expression for pages without `ArmEEGDemo`.

## Preview and checks

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory frontend
# http://127.0.0.1:8000/summary/preview.html       synthetic scenarios: completed, interrupted, empty, multisubject
# http://127.0.0.1:8000/?stream=1                   live mode; End Run / New Run appear next to Reset
node frontend/tests/session-summary.cjs            # metrics, adapter, metadata, view HTML
node frontend/tests/session-run.cjs                # lifecycle against the real Controller with a fake board
node frontend/tests/browser-summary.mjs            # isolated headless Chrome (port 9228), fake serial only
```

## Metric definitions

- Completed trial: one validated RESULT for one request (session nonce + request id). Unresolved, failed,
  timed-out and duplicate responses are never counted.
- Overall accuracy = correct completed / completed. Unavailable (not 0%) when there are none.
- Unique recording: distinct `sourceTrialId` (dataset id + metadata `trial_id`, which already encodes
  subject/session/run/cue). Unique-recording accuracy scores only the first completed result per recording.
- Repeated evaluations = completed − unique.
- Per-class accuracy and the confusion matrix (rows actual, columns predicted) include every completed trial,
  so repeats are counted; the panel says so.
- Elapsed session time = first Play → End Run, pauses included. Browser round trip = request creation →
  accepted RESULT in the page, including UART transfer; labeled as such and never called inference latency.
- Status: `completed`, `interrupted` (a request was still unresolved at End Run; it is named and excluded) or
  `empty`.

The visible card shows: the title and run status; accuracy (hero) with correct/incorrect, Trials,
Duration (“Includes pauses”); a “N unique recordings · M repeat evaluations” line; the Confusion Matrix
(rows = actual, columns = predicted, blue diagonal = correct, muted amber off-diagonal = incorrect, each
cell labelled by its ACTUAL class, e.g. “Correct left” / “Incorrect left”, from the metadata `short` names;
each actual-class row labelled with its accuracy or “Unavailable”); and Top EEG Channels. Model name, experiment, source, model hash, firmware label, session id, seed,
dataset, subject, browser round trip and the full channel list stay in the frozen snapshot
(`EdgeSessionRun.current.summary`, `history`) and in the capture export; they are not rendered. The
synthetic-preview label is shown only by `preview.html`, outside the card.

### Top EEG Channels (EEG signal overview)

Visible: subtitle “RMS (µV)”, up to three channels by descending pooled RMS on one zero-based linear scale,
and a compact “Prerecorded EEG · N unique recordings · k artifact-flagged” line. Equal RMS values are
ordered by channel name. `summary.eeg.channels` keeps every ranked channel.

Label “Prerecorded EEG · channel RMS”. For each channel, RMS = sqrt(Σ samples² / sample count) pooled over the
EEG windows of the **unique** completed recordings (a repeat evaluation never adds its window again). Samples
come from the dataset provider (`dataset.channels(index)`: filtered µV before normalization/quantization,
units verified by the model metadata `eeg` block). A channel is ranked only if it has finite, non-flat
samples in every recording that has EEG, so compared channels share one recording set; otherwise it is listed
as excluded. Fewer than two rankable channels, no EEG, or unverified units → “EEG amplitude unavailable ·
reason”. Artifact-flagged recordings are included (count shown). Missing channels are never zero-filled.
Accumulation happens once per unique recording as results arrive (including the final in-flight result);
nothing runs per animation frame. The note “Amplitude ranking does not indicate neural activation” is fixed.

## Adapter interface

```js
EdgeSessionSummary.build(session, results) -> frozen summary {session, classes, results, metrics}
session: {sessionId, modelId, modelName, experimentName, datasetId, datasetName, subjectId, modelHash,
          modelHashVerified, firmwareBuild, seed, source, status, interruption, note, startedAt, endedAt,
          classLabels /* required: {index: name|{name,display}} */, classNamesVerified}
result:  {trialId /* unique attempt */, sourceTrialId /* stable recording identity */, subjectId,
          groundTruth, predictedClass, logits, roundTripMs /* only when measured */}
EdgeSessionSummary.adaptStreamingRecords(controller.records, {datasetId, dataset, roundTrips}) -> results
EdgeSessionSummary.build(session, results, {eeg}) attaches an EdgeEEGOverview result as summary.eeg
EdgeEEGOverview.createAccumulator({units, unitsVerified}).add(sourceTrialId, channels, {artifact}) / .result()
EEG provider (attach option `eeg`, default built from dataset.channels + metadata.eeg):
  {units, unitsVerified, samples(record) -> {channel: samples[]} | null, artifact(record) -> bool | null}
```

Missing optional fields render as “Unavailable”. Class order and names come from `classLabels`; nothing
assumes class 0 is left. `model-metadata.js` is the only place that names a model, subject, dataset or class.

## Integration guide for `feature/multisubject-eeg-trials`

1. Add one entry to `EdgeModelMetadata.models` with the new PTE SHA-256, `modelId`, `modelName`,
   `experimentName`, `datasetId`/`datasetName`, `subjects` (all subjects the loader may serve), `sessions`,
   `classLabels` taken from that model's verified `class_map` (each entry `{name, display, short, order}`;
   `short` feeds the matrix cell labels and defaults to `display`; `order` is the display position so the
   matrix can show Left then Right even when the model's class indices are reversed, and defaults to the
   index; all counts are keyed by index), and an `eeg` block (`units`, `unitsVerified`,
   `source`) only if the dataset's EEG buffers are verified microvolts; without it the EEG overview says
   “units not verified” and the classification summary still opens. `resolve()` refuses a dataset whose
   trials fall outside `subjects`/`sessions` or whose `model_sha256` differs, so a mismatch shows as
   unverified instead of borrowing Model 2's names.
2. Keep records in the controller's shape (`session`, `request`, `trial`, `trial_id`, `ground_truth`,
   `class_id`, `logits`). Make `trial_id` unique across subjects (e.g. `S3/1test/run2/cue1234`). The
   dataset's `trials[].source_trial.subject` supplies `subjectId` per result; with several subjects the
   session-level subject shows “Unavailable” while each result keeps its own.
3. After your handshake succeeds, call `window.EdgeSessionRun?.attach({...})` exactly as in `bridge.js`,
   passing your controller, dataset, `firmware`, the same `cancelVisualization` the controller received, and
   your reset flow (which must call `controller.reset`). One controller per model: do not mix models in a
   session; a new handshake means a new attach.
4. The manager wraps `play`, `reset`, `visualize`, `onState` and `onRecord` on that controller instance.
   It needs `pause()`, `task`, `pending` (with `request` and `trial.trial_id`), `failed`, `ready`, `seed`,
   `session`, `records`, `archives`-style reset semantics, and the controller's existing refusal to reset
   an unresolved request. No change to `controller.js` or the UART protocol is required.

## Not covered

- Saved UART replay (legacy page) is not summarized; if added later it must use `source: 'saved-uart-replay'`
  and a separate controller so live and replay results never share a session.
- No model inference latency is reported; the firmware does not expose one.
