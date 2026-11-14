# Tuần 5 — thực thi readiness và domain adaptation, 14/11/2026

Phụ lục sau khi ngài cung cấp đường dẫn `dataset` và xác nhận link tác giả: [audit corpus local](WEEK5_DATASET_CANDIDATE_AUDIT_20261114.md). Đã tìm thấy ảnh/nhãn legacy, nhưng danh tính D-Fire của corpus local vẫn NOT VERIFIED; training tiếp tục BLOCKED. Các artifacts run v2 bên dưới được giữ nguyên để bảo toàn lịch sử.

BRANCH `fix-4`, HEAD `730342623b5946f0eb9d25241eb4b8fc20488e79`. **WEEK5_STATUS=BLOCKED_DATA_READINESS**. Đã audit dữ liệu thật, tạo dataset version/queues, kiểm baseline và parameter policies, triển khai runner có cổng dữ liệu/tài nguyên, chạy tests và ba CLI gate checks. **0 training runs, 0 epoch, 0 adapted checkpoint**. Chưa có baseline mAP trên scope mới, ΔmAP, đo quên hoặc selected model.

Artifacts phiên bản cuối: [run_20261114_73034262_v2](../runs/benchmarks/week5_domain_adaptation/run_20261114_73034262_v2/). Bản chuẩn bị v1 vẫn giữ local; PowerShell đã coi log stderr là NativeCommandError (outer exit 1 dù code tạo đủ artifacts). Đổi logging sang stdout và chạy lại hoàn toàn vào v2 mới: **exit 0**. Log `.tools/week45_closure_73034262/week5_prepare_v2.log`; không có training trong cả hai lần.

## D-Fire: dữ liệu chưa có

Đã đọc `data_dfire.yaml`, các data YAML, training args và Ultralytics configured datasets root; corpus D-Fire được khai báo ở `datasets/dfire`, không tồn tại. Rà workspace/configured roots lần đầu 411 thư mục; recheck công cụ public sau khi tạo artifacts: **422 thư mục, 0 lỗi, 0 root có images**. Giới hạn: rà workspace và các root cấu hình, không khẳng định đã rà mọi ổ đĩa/máy khác; bỏ cache/venv/browser và có deadline 120s.

| Đường dẫn đã kiểm | Tồn tại | Tổng images train/val/test |
|---|---|---:|
| `D:\comfyui\ComfyUI_windows_portable\datasets\D-Fire` | Không | 0 |
| `D:\comfyui\ComfyUI_windows_portable\datasets\dfire` | Không | 0 |
| `D:\workspace\D-Fire` | Không | 0 |
| `D:\workspace\DACN_NCKH\D-Fire` | Không | 0 |
| `D:\workspace\DACN_NCKH\datasets\D-Fire` | Không | 0 |
| `D:\workspace\DACN_NCKH\datasets\dfire` | Không | 0 |
| `D:\workspace\DACN_NCKH\dfire` | Không | 0 |
| `D:\workspace\dfire` | Không | 0 |

`WEEK5_DFIRE=BLOCKED_DATASET_MISSING`. Các checkpoint/args còn có thể xác thực SHA, nhưng không thay thế original corpus và split provenance. Không tạo D-Fire từ FireSmokeLabeler/legacy dataset. Cần original images + YOLO labels + split train/val/test và nguồn annotation/split để audit hashes/counts/class mapping/leakage. [data_availability_recheck.json](../runs/benchmarks/week5_domain_adaptation/run_20261114_73034262_v2/data_availability_recheck.json) và `dfire_availability.json` giữ receipt; chưa có content manifest D-Fire, field là null.

## Dataset Việt Nam: label đúng cú pháp chưa đủ là GT training

Đã kiểm **3.121 ảnh và label**: train 1.868 / val 623 / test 630, không missing, decode hoặc bbox syntax/bounds error. Mapping theo `data_vn.yaml`: 0=SMOKE, 1=FIRE. SHA ảnh/label hiện tại được ghi riêng; current source hashes được kiểm lại trên các nguồn đã fingerprint, nhưng original extractor không có runtime SHA/PTS để chứng minh từng legacy frame hiện trỏ chính xác tới ordinal/time nào.

Không có hash-bound bbox approvals trong các evidence đã kiểm; `review_status.csv` có boolean `is_reviewed=True`, thiếu người/time/evidence/signoff/source binding cho từng bbox. Điều này là thiếu bằng chứng, không có nghĩa mọi bbox đều sai. VN_POS mới có human **event intervals**, chưa phê duyệt bbox trên frame. `BBOX_GROUND_TRUTH_REQUIRED` vẫn áp dụng. Không dùng model prediction để tạo nhãn hoặc coi interval là box.

- **0** nhóm duplicate SHA/source/video băng qua legacy splits trong kiểm tra hiện tại; **75** cặp pixel similarity giữa các split là hypotheses cần review, không tự gọi là exact duplicate hay proof độc lập. `cross_split_pixel_candidates.csv` dẫn tới ảnh thực.
- **129 ảnh label rỗng từ 15 VN_NEG nonnegative/uncertain** có queue riêng. Có thể một video có khói nhưng nhiều frame khác chỉ là người dẫn/phỏng vấn; không tự coi 129 ảnh đều sai. Sáu contact sheet là ảnh thực không overlay model/GT, đã xem QA sheet đầu để kiểm bố cục, không coi là human annotation.
- `vn_image_candidates.csv` giữ SHA, source candidate, frame/time declaration, split reservation và trạng thái `approved_for_training=False` cho toàn bộ 3.121 ảnh. `bbox_annotation_queue.csv` có **1.904 ảnh** ở train/val reservations để tiếp tục bbox review; không đưa test candidates/blocked VN_POS vào training queue. Queue không cấp quyền training và không xác nhận scene independence.
- `empty_label_review_129.csv` bao gồm ảnh thuộc reservation lịch sử khác nhau; review từng ảnh được phép, không dùng test metrics hay chuyển test sang train.
- `dataset_manifest.json` là phiên bản **BLOCKED_NOT_A_TRAINABLE_DATASET**, approved rows=[] được ghi rõ có chủ đích. Không gọi đây là dataset train/val/test hợp lệ hoặc chạy YOLO trên empty scope. Nó hash-bind candidate CSV, prior split manifest và GT event scope.

Giữ reservation VN_POS 27/11/13/29; event-positive accepted theo nhóm 26/9/11/24. Scene/camera chưa có metadata, scene groups vẫn provisional. Scene review có thể tạo manifest phiên bản mới với prior SHA + reason; không sửa manifest đã hash, không chọn split theo AI predictions/performance.

## Baseline và môi trường đã kiểm thực tế

Checkpoint `runs/detect/runs/benchmarks/yolov8n_seed42/weights/best.pt` đúng SHA `f3631798de1cbc5742add6d81eadec055821ee59d7dec431fd9e71a377662eb6`, đọc được bằng Torch **2.15.0+cpu** / Ultralytics **8.4.198**, mapping 0 smoke/1 fire. Backbone topology thực có **10 modules**. Checkpoint bytes trước/sau không đổi. CUDA **false**, preflight dùng CPU; tổng RAM 16948453376 bytes, available tại preflight 5941796864 bytes; VRAM null. Không cài dependency mới.

Kiểm mask và optimizer trên baseline thực: full/replay **3.011.222 learnable parameters**, freeze **1.738.566**, trên tổng **3.011.238**; 16 DFL projection parameters cố định ở mọi strategy theo Ultralytics, không phải bỏ sót fine-tune. Optimizer chỉ giữ đúng các tensor trainable, coverage khớp. Digest frozen backbone gồm parameters **và BN buffers** được lưu trước training. Chưa huấn luyện, nên proof backbone không đổi sau training thật là **null**, không mượn synthetic test để kết luận thực nghiệm.

Resource estimate và baseline VN/D-Fire mAP đều **null** vì dữ liệu chưa đủ; CPU fallback có code nhưng cần timed pilot có dataset/checkpoint/device/log hashes và runtime budget. Không khởi chạy workload 30 epochs x 3 seeds x 3 strategies trên tập chưa hợp lệ.

## Config, runner và giao thức đã chuẩn bị

Giữ ba config plan cũ; config chung mới [week5_execution_protocol_v1.json](../configs/research/week5_execution_protocol_v1.json) khai báo trước: cùng baseline, mapping, scope VN, 640px, 30 epochs dự kiến, seeds 42/123/456, batch 8, SGD/lr0=.01/momentum=.937/weight_decay=.0005, deterministic, augmentation không trộn thêm image presentations (mosaic/mixup/copy-paste/cutmix=0). Seeds/budget có thể đổi chỉ qua protocol version mới trước thực nghiệm, không vì một strategy thắng/thua.

Runner [run_week5_domain_adaptation.py](../scripts/run_week5_domain_adaptation.py):

1. Fail closed trước NN nếu manifest chưa READY, bbox/source/scene lineage hoặc grouped-split/leakage signoff thiếu. Hash-check ảnh/label/source/native-frame receipt; không đưa reserved tests vào train/validation staging.
2. Cần timed pilot + budget receipt thật; chưa có estimate thì dừng. Dữ liệu approved được copy sang run mới, originals không ghi đè; khóa manifest/config/budget/staged hashes trước training.
3. Đo baseline validation VN và D-Fire khi dữ liệu tương ứng đủ, trước train; chưa dùng test để chọn checkpoint/model. Full/freeze có thể chạy VN hợp lệ khi D-Fire còn thiếu, nhưng retention/so sánh ba strategy vẫn thiếu.
4. Full fine-tune mở mọi learnable tensor trừ fixed DFL; freeze đọc topology thực, loại frozen tensors khỏi optimizer trước resume state load, ép backbone eval trước batch để BN không đổi, đồng bộ frozen EMA state, kiểm digest model và checkpoint đã lưu sau training.
5. Replay chỉ D-Fire **TRAIN**: chọn seeded không hoàn lại `ceil(V/4)` ảnh cho V ảnh VN; cùng selection mỗi epoch, mỗi ảnh xuất hiện một lần. Ghi IDs/hashes/seed/fraction/sai lệch làm tròn, kiểm actual image presentations từng epoch. Không lấy D-Fire val/test replay.
6. Ghi epochs thực, validation history, LR/optimizer/parameter count, last/best SHA, runtime/device/error/recovery. Resume chỉ nhận receipt/hash đúng và checkpoint có optimizer/epoch thực, không nhận final stripped checkpoint giả là resumable.

**Phạm vi kiểm chứng runner:** fail-closed CLI thật + unit tests cơ chế; nhánh huấn luyện/evaluation/resume trên dataset thật **chưa được chạy** vì blockers. Video inference mới cho adapted checkpoint **chưa chạy**; `verify_video_comparison` mới kiểm contract scope/GT/sampling/temporal/evaluator tương ứng. Không tuyên bố có video re-evaluation hoặc best model từ các contract này.

Ba gate checks thực cho full/freeze/replay seed42 đều exit **2** theo thiết kế, receipt `training_executed=false`, epochs=0, không checkpoint. Đây là kiểm tra cổng, không phải ba training runs. [readiness_gate_checks.json](../runs/benchmarks/week5_domain_adaptation/run_20261114_73034262_v2/readiness_gate_checks.json) giữ lệnh, elapsed, log và exit thực.

## mAP, delta và độ quên

| Strategy | Seed | VN mAP50 | VN mAP50:95 | D-Fire mAP50 | D-Fire mAP50:95 | Delta VN | Delta D-Fire | Event Recall | FAR/hour | Delay | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 42 | null | null | null | null | null | null | null | null | null | BLOCKED_EVALUATION_DATA |
| full_finetune | 42 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| full_finetune | 123 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| full_finetune | 456 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| freeze_backbone | 42 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| freeze_backbone | 123 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| freeze_backbone | 456 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| replay_20_percent | 42 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| replay_20_percent | 123 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |
| replay_20_percent | 456 | null | null | null | null | null | null | null | null | null | NOT_TRAINED_DATA_BLOCKED |

Per-class FIRE/SMOKE AP50/AP50:95, cả hai ΔmAP và catastrophic forgetting cũng **null**, không giả thành 0. Không lấy old result curves/test mAP làm baseline trên scope mới. `experiment_matrix.json/csv` lưu đủ trường; mean/std=null, measured seeds=0. Δ = adapted−baseline; D-Fire delta âm chỉ được diễn giải là suy giảm khi đã có cùng evaluation scope và số đo hợp lệ.

Model selection đã khai báo ở protocol và kiểm bằng tests: validation-only, đủ VN+D-Fire mAP và paired video/retention/runtime evidence; video recall và event-instance recall không giảm quá .05, FAR không tăng, mean matched delay <=10s. Xếp theo VN mAP50:95, D-Fire mAP50:95, recall, FAR, delay, SHA. Đây là ràng buộc prospective mới, không đổi target/kết quả Week4 đã niêm phong. Hiện **selected_model=null**; không dùng image mAP đơn lẻ chọn best. Sau khi có adapted checkpoint, còn phải chạy video validation cùng source-PTS 5FPS, GT/scope/temporal/evaluator; không thay historical exploratory metrics cho run mới. Test chỉ dùng sau selection/lock/independence đủ bằng chứng.

## Bbox/source và resource receipt cần cung cấp

Manifest READY thật phải lưu riêng khi có danh tính người review; chỉ công bố provenance/hash đã khử định danh. Các trường mỗi ảnh VN gồm `corpus=VN`, split, image/label path+SHA, source_video_path+SHA, scene_group_id, `lineage_verified=true`, source_frame_index, source_pts_ticks, time_base và native lineage receipt path+SHA. Receipt native decode phải bind đúng image/source/frame/PTS và status VERIFIED_NATIVE_DECODE, không phải boolean tự đặt từ FPS.

`bbox_review` cần `status=SUBMITTED_BBOX_REVIEW`, `attest_frame_review=true`, reviewer_id thật, reviewed_at_utc thật, evidence_ref và image/label/source hashes khớp; uncertainty không dùng làm training GT. Ảnh negative cần người xác nhận trên đúng ảnh, không suy từ video decision. Đây là schema cần có, **không phải signoff đã tạo**. D-Fire dùng original annotation origin + official split provenance SHA/source groups, không bắt người dùng gán lại corpus gốc.

Resource receipt phải có dataset_sha256, baseline_sha256, device (`cpu` hoặc `0`), measured_epoch_seconds từ pilot thật, max_training_hours, pilot_log+SHA. Không điền số giả để qua gate. Recheck bằng `validate_budget`, có headroom 25% so với epochs x thời gian đo.

## Blockers: nguyên nhân, ảnh hưởng, cách sửa và kiểm lại

| Blocker | Ảnh hưởng | Cách sửa | Kiểm lại |
|---|---|---|---|
| D-Fire missing | Replay/retention/baseline D-Fire mAP chưa làm được | Cung cấp corpus/archive gốc + split/labels provenance | Availability audit, content hashes, class/split/near-duplicate audit |
| Bbox GT chưa đủ provenance | VN baseline và cả ba train strategy bị chặn | Export bbox review thật hoặc bổ sung evidence cho labels sẵn có; review 129 ảnh từng ảnh | Hash-bound per-frame signoff, bounds/classes/uncertain checks |
| Legacy frame lineage chưa xác thực | Không biết chắc source frame/time của ảnh cũ | Kiểm pixel/native ordinal/PTS hoặc re-extract version mới từ nguồn checksum đúng | Native decode receipt bind image/source/frame/PTS |
| Scene relations unresolved | Clean split/independent test chưa chứng minh | Review 75 cặp và prior scene groups, giữ reservation; version nếu đổi | Disjoint source/scene/image groups; unresolved groups quarantine |
| CPU-only, chưa timed pilot | Không có runtime estimate an toàn | Pilot sau khi có data hợp lệ trên thiết bị dự kiến | Log SHA/device/budget/estimated bound |

Chi tiết máy đọc: [blockers.json](../runs/benchmarks/week5_domain_adaptation/run_20261114_73034262_v2/blockers.json). Không yêu cầu làm lại 79 event reviews; bbox/frame review là tác vụ khác với GT sự kiện.

## Lệnh tái lập và kiểm chứng

```powershell
# Mọi output dưới đây phải mới; không ghi đè run đã có.
python scripts/audit_week5_data_availability.py --output runs/benchmarks/week5_domain_adaptation/NEW_VERSION/data_discovery.json
python scripts/prepare_week5_domain_adaptation.py --output-dir runs/benchmarks/week5_domain_adaptation/NEW_PREPARED_VERSION --discovery-receipt runs/benchmarks/week5_domain_adaptation/NEW_VERSION/data_discovery.json --closure-dir runs/benchmarks/week4_acceptance_closure/run_20261114_73034262
# Manifest hiện tại sẽ dừng exit2; READY_DATASET.json chỉ được dùng khi có dữ liệu và evidence thật.
python scripts/run_week5_domain_adaptation.py --dataset-manifest READY_DATASET.json --strategy full_finetune --seed 42 --budget-receipt MEASURED_PILOT.json --output-dir runs/benchmarks/week5_domain_adaptation/NEW_FULL_RUN
# Đổi --strategy thành freeze_backbone hoặc replay_20_percent trong output mới.
# --resume chỉ dùng với receipt INTERRUPTED/RUNNING và last.pt có optimizer thực.
```

Targeted pytest: **119 passed, 58 subtests passed**, exit **0**, thời gian wall **22.441s**. Full unittest: **350 tests, OK**, exit **0**, thời gian framework **148.664s**, wall **152.649s**. Không failed/skipped. Các fixture stream hỏng có thể in cảnh báo codec; không phải lỗi nguồn thực nghiệm. Report RTSP cũ được khôi phục đúng bytes trong `finally`.

Lệnh thực tế, timestamps, count, SHA log: [test_results.json](../runs/benchmarks/week4_acceptance_closure/run_20261114_73034262/test_results.json). Raw logs local: `.tools/week45_closure_73034262/targeted_final.log`, `.tools/week45_closure_73034262/full_unittest_final.log`; không commit raw logs.

Synthetic tests chứng minh cơ chế freeze/BN/optimizer, replay ratios/train-only/determinism, bbox/leakage/source binding, resume contracts, AP aggregation/per-class/null/delta, paired video constraints và selection; không là bằng chứng metrics hoặc frozen checkpoint sau training thực. Các test/công cụ chỉ chạy Windows local; không tự khai PASS Linux/CI.

Không commit/push/xóa nguồn, checkpoint, private GT, submitted review hoặc raw data lớn. 416 raw artifacts lịch sử đang chờ authorization cleanup vẫn giữ nguyên. Những artifacts local preview/weights/cache/logs không được đưa vào whitelist Git.
