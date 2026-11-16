# Week 5 final deliverable — PARTIAL / BLOCKED

Branch `fix-4`, HEAD kiểm tra `8375350d6127040cdd6cfe593c606f0fe718cef1`;
worktree sạch trước triển khai, có thay đổi code/tests/tài liệu khi bàn giao.
Không commit/push, sửa nhãn, thay checkpoint, sửa private storage hoặc frozen GT.

Sản phẩm local: `runs/week5_domain_adaptation/final/`. Đây là bàn giao chuẩn bị
và pipeline kiểm chứng, **chưa đạt nghiệm thu thực nghiệm Tuần 5**. Không có run
baseline mới hoặc A/B/C hoàn tất; không có checkpoint mới hoặc mô hình thắng.

## Dữ liệu và blocker thực tế

Đã quét read-only private storage, đối chiếu queue và chạy strict committed review:
1.922 ảnh, 1.919 APPROVE, 3 REJECT, 0 pending; 1.943 annotation versions hợp lệ,
21 version lịch sử bổ sung; 5 quarantine đã giải quyết, 0 unresolved.
Receipt: `final/execution_logs/live_review_verification.json`.
Không đồng nhất review hoàn tất với dataset eligible.

Snapshot đã có: 1.258 negative, 256 SMOKE-only, 102 FIRE-only, 303 cả hai;
675 smoke boxes, 874 fire boxes. Reservation APPROVE: train 1.376, val 525,
test 18. Chúng chưa phải tập training được phép sử dụng.
Lineage trong snapshot: 797 verified derived, 1.125 unresolved legacy.

Active eligible VN train/val/test = **0/0/0**, D-Fire = **0/0/0**.
VN thiếu quan hệ scene đối lập đủ chứng cứ theo protocol đang khóa.
D-Fire còn 750/751 suspected groups chưa phân xử; một nhóm đã xác nhận overlap.
Không coi cả 751 nhóm là duplicate. Candidate corpus còn nhãn lỗi và có vấn đề
độc lập scene; không dùng YAML cũ để huấn luyện.
Hai candidate VN 85 hoặc 110 ảnh đã khảo sát thiếu FIRE ở validation, không đủ
so sánh hai lớp chính thức. Không dùng ngưỡng/mAP/test để lựa subset.

URL/GPS/camera ID không phải trường bắt buộc của đề cương; thiếu các trường này
không tự tạo blocker. Tuy vậy, absence of duplicate matches không chứng minh
độc lập tuyệt đối. Quan sát của người dùng trên 12 contact sheets đã lưu trong
audit trước; không yêu cầu xem lại hoặc ký hàng loạt. Proposal v3 vẫn chưa active;
không có thay đổi gate/protocol để hợp thức hóa training.

## Baseline và pipeline

Baseline cũ: `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt`,
SHA `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`.
Membership chưa xác minh; **không sử dụng làm baseline so sánh chính thức**.
Initialization dự kiến `yolov8n.pt`, SHA
`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`;
hash local đã kiểm tra, chưa phục hồi receipt download gốc.

- `scripts/run_week5_fresh_baseline.py`: baseline D-Fire TRAIN-only, khóa SHA
  data/protocol/init/pilot, xác minh ảnh/nhãn, không đưa test vào trainer;
  ghi image presentations thực qua batch loader, kiểm từng epoch, class mapping,
  checkpoint/validation SHA. Dừng BLOCKED trước model load nếu dữ liệu chưa đủ.
- Runner A/B/C hiện có được giữ: full fine-tune, freeze backbone, freeze + replay
  ceil(V/4) D-Fire TRAIN duy nhất, fixed-seed, không replacement. Tỷ lệ thực là
  ceil(V/4)/(V+ceil(V/4)); có rounding, không diễn giải thành 20% toàn corpus.
- `src/week5_comparison.py`: kiểm đủ 3 strategy × 3 seeds, cùng baseline SHA,
  dataset và evaluator/config SHA, đủ epochs, metrics hợp lệ; tính delta có dấu,
  forgetting=max(0,-delta_DFire). Thiếu lớp giữ AP và aggregate chưa đủ hai lớp
  là null. CPU fallback có kiểm tra và có thể bị tắt rõ ràng.
- `scripts/compare_week5_runs.py`: CLI thu receipt thực, xác minh checkpoint,
  membership baseline D-Fire TRAIN, staged image/label SHA, tính class support
  từ GT validation đã xác minh. Không đọc test metrics; không tự xuất winner.
- `select_final_model`: validation-only, cả hai lớp, mục tiêu MT4 VN map50 +0,05
  và D-Fire không giảm quá 0,02, đồng thời giữ acceptance video/runtime v2.
  Chưa có evidence đầu vào thật, nên **NO_ACCEPTED_WINNER**.

Các đường training mới đã kiểm chứng preflight bị chặn; **chưa kiểm chứng bằng
một training run thành công**. API comparison/export được test bằng dữ liệu
synthetic, không phải số liệu thực nghiệm. Tính đúng trên GPU và runtime/resource
pilot còn cần đo sau khi dataset READY. Không báo pipeline mới đã được nghiệm
thu end-to-end. Quy trình final winner vẫn cần paired validation video evidence
theo v2, decision được kiểm tra và checkpoint hash thực trước xuất alias.

## Bảng thực nghiệm

| Model | VN mAP50 | VN mAP50:95 | D-Fire mAP50 | D-Fire mAP50:95 | ΔVN | Forgetting |
|---|---|---|---|---|---|---|
| Baseline mới | chưa đo | chưa đo | chưa đo | chưa đo | — | — |
| A | chưa đo | chưa đo | chưa đo | chưa đo | — | — |
| B | chưa đo | chưa đo | chưa đo | chưa đo | — | — |
| C | chưa đo | chưa đo | chưa đo | chưa đo | — | — |

`comparison.csv/json` có schema cả per-class AP50/AP50:95 và support;
metric null/CSV rỗng, không thay bằng 0. Hai PNG ghi NOT MEASURED, không phải đồ
thị kết quả. `checkpoint_sha256.json` chỉ hash weights có thật và exported={}.
Không tạo bất kỳ `baseline_best.pt`, `strategy_*_best.pt`, `best_adapted.pt` giả.

## Lệnh tái lập

Lệnh thực tế, exit code, thời gian và SHA log: `final/execution_logs/tests.json`.
Preflight baseline và từng A/B/C trả **2**, receipt xác nhận training_executed=false,
epochs_actually_completed=0. Đây là kiểm chứng gate, không phải training thất bại
giữa chừng. Lệnh chuẩn bị cũng trả 2 khi bàn giao BLOCKED thành công.

```powershell
python scripts/finalize_week5_deliverable.py --evidence runs/benchmarks/week5_domain_adaptation/final_bbox_20261116_57d1b826 --output-dir runs/week5_domain_adaptation/final_reproduction_NEW
python -m pytest -q tests/test_week5_comparison.py tests/test_domain_adaptation.py tests/test_scene_subset.py tests/test_scene_evidence_requirements.py tests/test_bbox_final_audit.py tests/test_bbox_resolution.py
python -m pytest -q tests
```

Đầu ra luôn phải là directory mới. Lệnh training dưới đây là template **chỉ sau
khi có manifest READY, protocol mới hash-bound và pilot thật**; các placeholder
chưa tồn tại, không phải lệnh được phép chạy ngay trên dữ liệu hiện tại:

```text
python scripts/run_week5_fresh_baseline.py --dataset-manifest READY_DFIRE.json --protocol LOCKED_FRESH_BASELINE.json --budget-receipt MEASURED_BASELINE_PILOT.json --seed 42 --output-dir runs/benchmarks/week5_domain_adaptation/fresh_baseline_NEW
python scripts/run_week5_domain_adaptation.py --dataset-manifest READY_VN_DFIRE.json --protocol LOCKED_NEW_BASELINE_ADAPTATION.json --strategy full_finetune --seed 42 --budget-receipt MEASURED_ADAPTATION_PILOT.json --output-dir runs/benchmarks/week5_domain_adaptation/A_seed42_NEW
python scripts/compare_week5_runs.py --dataset-manifest READY_VN_DFIRE.json --protocol LOCKED_NEW_BASELINE_ADAPTATION.json --baseline-receipt VERIFIED_BASELINE_RECEIPT.json --receipts ALL_NINE_TRAINING_RECEIPTS --output-dir runs/week5_domain_adaptation/measured_comparison_NEW
```

Lặp runner cho freeze_backbone, replay_20_percent và seeds 42/123/456. Không trỏ
protocol hiện tại sang baseline mới bằng sửa checksum tùy tiện: tạo protocol kế
thừa mới với checkpoint thực, SHA, parent và membership đã khóa.

## Việc tiếp theo

Người dùng không cần review lại 1.922 ảnh hoặc 12 contact sheets. Cần quyết định
có chấp thuận thay đổi evidence protocol v3 đã đề xuất hay giữ v2; proposal không
tự active. Ngay cả khi chấp thuận, không tự biến UNKNOWN thành independent.
Codex tiếp tục phân xử/exclude source/scene chưa rõ bằng evidence, bảo vệ test,
đạt validation hai lớp và D-Fire READY, rồi đo pilot trước baseline/A/B/C.
Kết quả tests cuối cùng được bổ sung ở receipt acceptance riêng, kèm SHA logs.


## K?t qu? ki?m ch?ng cu?i

- Targeted: **72 passed, 37 subtests**, exit **0**.
- Full: **467 passed, 80 subtests**, exit **0**, 168,01 gi?y pytest.
- Baseline v? A/B/C preflight: exit **2** ??ng k? v?ng, training_executed=false, epochs=0.
- L??t full ??u: exit **1**, 7 failed / 457 passed; nguy?n nh?n t?i s?a module trong khi pytest ?? import b?n tr??c. Gi? nguy?n log c?; ch?y l?i tr?n m? c? ??nh ?? PASS.
- Receipt chu?n cu?i: `execution_logs/verification_v2/tests.json`, g?m command/exit code/SHA log.
- Full log SHA256: `4e0976ac7fd84c9ceef58eab011d00dc2c427d87595cf376b2b3182d434ead91`.
- Targeted log SHA256: `3ceb415a8876b4151c7f36a348e299230bf6bb3098a42db9731c65e9622d1000`.
- ?? x?c minh 17/17 parent seal artifacts; strict private review 1.943 versions h?p l?, 1.919 approve / 3 reject / 0 pending; 5 resolved / 0 unresolved.
- Frozen events.jsonl v? events_v1.1_snapshot.jsonl v?n SHA `8c6f30bbd057d5b5ae453feb64eb010387c2087ab9350ccea922ba04325bf7c3`.
- PyTorch 2.15.0+cpu, CUDA=false ch? l? th?ng tin m?i tr??ng; **ch?a c? resource pilot**.
- Test PASS x?c nh?n contracts/pipeline, kh?ng thay th? th?c nghi?m scientific acceptance. **Week 5 PARTIAL/BLOCKED**.
