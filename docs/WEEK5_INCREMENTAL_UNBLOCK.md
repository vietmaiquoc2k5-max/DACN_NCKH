# Week 5 incremental unblock — 15/11/2026

**PARTIAL: lineage expansion is still running in the background.** This report freezes 644 receipts (356 verified, 288 unresolved), not all 1,793 newly requested images. Remaining planned images are NOT YET AUDITED, not automatically unresolved. Live log: `.tools/week5_unblock_2e4b73a5/lineage.log`. The running command writes only new local receipts and does not update this frozen report/export. Full pytest collected before the newly added membership test: 380 passed; that additional test ran separately and passed (1 passed). No claim of full completion.

Branch fix-4, HEAD 2e4b73a50055fd1f80fcbc735a720dadf9f34dd0. Worktree initially clean; changes uncommitted. No commit/push/source deletion/checkpoint replacement.

## Human review and lineage

Snapshot: {"snapshot_id": "unblock_20261115_2e4b73a5", "snapshot_receipt_sha256": "214df2fe91e2795cc036d1af004568d41a6cd72fb6c9d09459d397554d2d6f41", "versions": 318, "unique_reviewed": 317, "approved": 317, "pending": 1605, "rejected": 0, "negative": 143, "positive_bbox": 174, "approved_delta": 210, "lineage_statuses": {"VERIFIED_DERIVED_IMAGE": 356, "UNVERIFIED_LEGACY": 288}, "approval_exclusion_statuses": {"APPROVED_PENDING_LINEAGE": 158, "APPROVED_LINEAGE_VERIFIED_PENDING_SPLIT": 159, "LINEAGE_VERIFIED_PENDING_HUMAN_APPROVAL": 197}, "eligible_train": 0, "eligible_val": 0, "private_export": "D:\\workspace\\DACN_NCKH\\.tools\\week5_unblock_2e4b73a5\\export"}

Live review remains operational. SQLite backup and all captured immutable version checksums verified. The snapshot is a fixed set; current live counts can increase. Previous snapshot had 107 approvals. No annotation generated or changed by agent.

Lineage expansion uses one decoder, exact documented resize/JPEG95 reconstruction and integer native PTS, no model similarity approval. Historical anonymization lacks per-image transform logs; nonmatching images remain UNVERIFIED. Previous 51 verified /78 unresolved receipts reused unchanged. No relabel requested merely for storage changes.

## Scene and membership

75 VN cross-split pairs checked for exact decoded pixels and contact sheets generated; {"UNRESOLVED": 75}. No independent scene proof inferred from non-equality. 751 D-Fire prior exact-pixel/group audit reused, still unresolved at automatic classification; prior visual same-camera evidence retained. Human queue includes IDs, source members, contact sheets, priority and evidence requirements. Test reservations preserved. No subset selected by performance.

Checkpoint metadata read: train_args.data=data_dfire.yaml, epoch=-1; current YAML refers to absent datasets/dfire. No original image-SHA training manifest/cache recovered from searched training/dataset paths. Baseline membership NOT_AUTHENTICATED. Proposed fallback in baseline_membership.json: lock scene-disjoint split, record initialization SHA and retrain new baseline on train only, validation selection, sealed test; A/B/C share that baseline. Not executed while data gates blocked.

Eligible train=0, validation=0. DFIRE_READY/VN_BBOX_READY/SPLIT_READY/RESOURCE_PILOT_READY/WEEK5_TRAINING_READY=false. Checkpoint SHA ready. Pilot/training/mAP/video metrics not executed. Strategy C remains freeze backbone + train-only replay20, ceil(V/4) denominator V+ceil(V/4).

## Tests

[
  {
    "name": "targeted",
    "exit_code": 0,
    "log": "D:\\workspace\\DACN_NCKH\\.tools\\week5_unblock_2e4b73a5\\targeted.log",
    "sha256": "ed2a51fddc51b800c6a3a87fa9e41ddb88d166745627cd937e877579e4970ef7",
    "tail": [
      "\u0000",
      "\u0000"
    ]
  },
  {
    "name": "membership_test",
    "exit_code": 0,
    "log": "D:\\workspace\\DACN_NCKH\\.tools\\week5_unblock_2e4b73a5\\membership_test.log",
    "sha256": "dea6cee3f94f278f1d8b9f4df11ef03a0c6a52be41e94ea5a68448c9988c28d9",
    "tail": [
      "\u0000",
      "\u0000"
    ]
  },
  {
    "name": "full",
    "exit_code": 0,
    "log": "D:\\workspace\\DACN_NCKH\\.tools\\week5_unblock_2e4b73a5\\full.log",
    "sha256": "fc67b06d233dfe1e316f079749514711954caf7b9201ddc2e3c7c5fb7042ebc0",
    "tail": [
      "\u0000",
      "\u0000"
    ]
  }
]

## Reproduce and next actions

`python scripts/recover_week5_frame_lineage.py --input .tools/week5_unblock_2e4b73a5/lineage_scope.csv --output NEW_OUTPUT --workers 1`

`python -m pytest -q tests/test_week5_readiness.py tests/test_bbox_qa.py tests/test_domain_adaptation.py tests/test_week5_unblock.py`

`python -m pytest -q tests`

Human: continue existing bbox workflow; resolve prioritized VN/D-Fire scene hypotheses with source/camera evidence. No repeated approval needed for unchanged annotations. Codex: join new snapshot approvals with receipts, validate grouped subset when evidence exists, then pilot/baseline/adaptation. Status PARTIAL/BLOCKED, not scientific acceptance.

Artifacts: runs/benchmarks/week5_domain_adaptation/unblock_20261115_2e4b73a5. Private snapshot/export/logs/contact sheets remain in ignored .tools/week5_unblock_2e4b73a5.
