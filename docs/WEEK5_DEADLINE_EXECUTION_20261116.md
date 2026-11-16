# Week 5 deadline — executed source review workflow

16/11/2026; fix-4 HEAD965d3e81be4e98022a0b5292a40c7f36e0291cd0. Worktree initially clean. Status **PARTIAL / TRAINING BLOCKED**. No original data/GT/approvals modified; no commit/push/delete.

## Executed work

Ranked **83** approved+verified-lineage train/val source groups, excluding REJECT, unverified lineage, VN_NEG_10 and reserved test. Fixed ranking uses class coverage, counts, diversity and SHA, never mAP. Generated6actual contact sheets (first/last verified frames), source paths/SHA, reservations and composition; inspected all6 and kept UNKNOWN because representative frames cannot establish whole-source independence, especially edited TV footage.

[
  {
    "pair": 1,
    "sources": [
      "VN_POS_13",
      "VN_POS_08"
    ],
    "potential_images": 85,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_01.jpg",
        "sha256": "d167c27003e962f4f54650b2aa30a75dfed516e3c6ea9fa3ff10ddc90a5245e3"
      }
    ],
    "decision": "UNKNOWN"
  },
  {
    "pair": 2,
    "sources": [
      "VN_POS_02",
      "VN_POS_08"
    ],
    "potential_images": 85,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_02.jpg",
        "sha256": "89772bf28c7ed3c4295c3ba94ed9fb2d3d141239f2b1229dd5ae6df58124377c"
      }
    ],
    "decision": "UNKNOWN"
  },
  {
    "pair": 3,
    "sources": [
      "VN_POS_14",
      "VN_POS_08"
    ],
    "potential_images": 83,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_03.jpg",
        "sha256": "3d19eef311dddd45971d98bcffa26a1a8d3300584bd24675640c18f040b138c4"
      }
    ],
    "decision": "UNKNOWN"
  },
  {
    "pair": 4,
    "sources": [
      "VN_NEG_01",
      "VN_POS_17"
    ],
    "potential_images": 80,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_04.jpg",
        "sha256": "ffaf086f15e7ea983a156863b8084c785f622395e5695233b833bd6e93fab6ec"
      }
    ],
    "decision": "UNKNOWN"
  },
  {
    "pair": 5,
    "sources": [
      "VN_POS_11",
      "VN_POS_08"
    ],
    "potential_images": 79,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_05.jpg",
        "sha256": "a00e5bec78636b75920752806f2e3fa6fe0d81d1b94ee0364c9f9f93bcbda348"
      }
    ],
    "decision": "UNKNOWN"
  },
  {
    "pair": 6,
    "sources": [
      "VN_NEG_01",
      "VN_POS_58"
    ],
    "potential_images": 76,
    "evidence": [
      {
        "kind": "VISUAL_SCENE_REVIEW",
        "path": ".tools/week5_deadline_20261116/pair_06.jpg",
        "sha256": "d68fc517dddf1cf846e954fd56b3c99f0d62b7e485064ec441710057bf26806d"
      }
    ],
    "decision": "UNKNOWN"
  }
]

Not asking to decide1380relations. Start with one or two pairs where authentic source/camera/context is actually known. Unused uncertain groups can remain excluded. No human decisions or signoffs prefilled. Empty initial batch has accepted_decisions0; initial_gate_verified receipt actor_idnull. Prior initial_gate_check is retained as an automated empty diagnostic, not a human decision.

## How to advance

Open human_scene_review.html in this bundle; enter actual Studio reviewer ID, inspect source footage and source metadata plus contact sheets. Leave UNKNOWN if source independence cannot be proved. Choose CONFIRMED_INDEPENDENT or CONFIRMED_OVERLAP only with evidence and reason; download JSON. In Studio BBoxQA, open **Week5 nhập quyết định source/scene đã review**, upload JSON, tick personal evidence attestation, and import. Auth uses current server-side Studio session; changing reviewer_id in JSON cannot impersonate another account. Token/password never entered into HTML or logs.

Importer checks exact pair scope, contextSHA, reviewer/time, evidence-file SHA, source-video SHA, duplicates/conflicts, parent/image/annotation/lineage hashes. Each batch creates a new UUID directory under decision_batches with receipt/checksum and fresh subset/readiness/leakage audit. Existing decision batch integrity/context checked before reuse. Never overwrite originals, snapshots or decisions. Need additional authoritative evidence? Keep UNKNOWN and obtain source/camera/footage origin; no reviewer should infer identity from pHash or two images alone.

build_subset requires all retained opposing pairs evidenced and maintains reservations. Import calculates exact train/val counts. **Nonempty is only PILOT_CANDIDATE_PENDING_PROTOCOL_VALIDATION**: research sample-size/ontology/provenance and D-Fire gates remain separate. It never starts training.

## Current result and baseline

No genuine source independence decisions supplied during agent run. Eligible train0/val0. Validated queue reference1919APPROVE/3REJECT; prior snapshot/source proof chain preserved. No class/source distribution claimed for eligible rows. Candidate composition per source is source_priority_report.json. Test samples excluded from ranking/selection.

D-Fire prior1confirmed overlap/750unresolved groups remain quarantined/candidate; no source-independent valid D-Fire train/val scope established. Baseline original image-SHA membership NOT_VERIFIED; baseline_protocol.json binds initialization f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 and class mapping through existing protocol, but split membership remains conditional. Need valid locked D-FireTRAIN/validation, membership/presentation logging and common baseline for A/B/C. C freezes backbone plus20%train-only replay.

Resource pilot NOT RUN because no qualified independent train/validation and D-Fire scope. Actual training0runs/0epochs/0new checkpoints; all experimental metrics null. No clean test/forgetting claim. Deadline does not change scientific gates.

## Tests and commands

[
  {
    "name": "targeted",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_scene_decision_import.py",
      "tests/test_bbox_final_audit.py",
      "tests/test_scene_subset.py",
      "tests/test_domain_adaptation.py",
      "tests/test_bbox_resolution.py",
      "tests/test_bbox_qa.py",
      "tests/test_gt_studio_ui.py"
    ],
    "exit_code": 0,
    "duration_seconds": 23.07655800000066,
    "log": ".tools/week5_deadline_20261116/targeted.log",
    "tail": [
      ".........................................................                                             [100%]",
      "57 passed, 43 subtests passed in 21.46s"
    ],
    "log_sha256": "a36c9eef5349b776847804cc19b0945f41caf4f02bfddff9f2c4eba6e5ffee99",
    "pytest_summary": [
      "57 passed, 43 subtests passed in 21.46s"
    ]
  },
  {
    "name": "full",
    "command": [
      "C:\\Users\\thien\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests"
    ],
    "exit_code": 0,
    "duration_seconds": 179.14633089999916,
    "log": ".tools/week5_deadline_20261116/full.log",
    "tail": [
      "[mjpeg @ 0000023342706480] overread 8",
      "[mov,mp4,m4a,3gp,3g2,mj2 @ 0000023343a34880] moov atom not found"
    ],
    "log_sha256": "4869113947ee2df784bb549e0a6602e2f109b24bea48253c003a301bfed96063",
    "pytest_summary": [
      "432 passed, 80 subtests passed in 177.49s (0:02:57)"
    ]
  }
]

Prepare a fresh bundle: `python scripts/prepare_week5_deadline.py --output .tools/NEW_DEADLINE_BUNDLE`. Studio imports currently point to .tools/week5_deadline_20261116. `advance()` can also be invoked by authenticated server workflow; do not fabricate actor dictionaries for real decisions. Tests use synthetic actors/evidence only. Full suite pytest commands/logSHA above.

The old bbox audit and original-source/label SHA manifests are reused, not relabeled. Import is an internal authenticated workflow, not a cryptographic signature against filesystem administrators. Human decision truth/authority must be supported by actual source/context; authenticated clicking alone cannot make false independence scientifically valid.

NEXT HUMAN ACTION: inspect6pair review page; decide only pairs with actual whole-source/camera evidence, upload through Studio. No BBox relabel/reapproval needed. NEXT CODEX ACTION: consume authenticated batch, validate all selected relations, formal sample-size/ontology and D-Fire; only then pilot/baseline/A/B/C. Current output is runnable fallback, not another claimed finished experiment.

Additional importer integration: 8 passed, exit0, log import_integration.log; added after full-suite collection and executed separately on final code. Authoritative fixed-artifact seal: delivery_seal.json. final_seal.json is a provisional generator index that included its live console log; do not use it as the final seal.
