# Tuần 6 — kế hoạch, chưa có nghiệm thu

STATUS: PLANNED. configs/research/week6_experiment_plan.json chứa9 cấu hình one-factor
imgsz416/640/960, confidence.25/.35/.5, K/N2/8,3/8,4/10. Baseline đề xuất640,.35,3/8;
không phải OP chính thức. Day/night cần metadata người kiểm định, không suy từ tên.
Phải khử cấu hình baseline trùng khi lập execution run; mọi run lưu config hash,
source/checkpoint/GT/split, exclusion zones và exact native-PTS sampling provenance.

Active Python: PyTorch có cài, CUDA unavailable; ONNX Runtime/OpenVINO/TensorRT
không được tìm thấy. Trạng thái backend installed không phải export/inference PASS.
Không tự cài dependency hoặc export model. Backend thiếu: NOT_TESTED_ENVIRONMENT_UNAVAILABLE.
CPU/GPU FPS và latency đều null, không dùng mô phỏng thay GPU thực.

Protocol chuẩn bị: warmup50 frame, measurement1000 frame, batch1, camera1/4/9;
lưu CPU/GPU/runtime versions, native/input resolution, throughput, per-frame và
end-to-end latency, RAM/VRAM, source timeline. GPU phải synchronize khi đo latency.
Đo riêng decoder, preprocessing, inference, tracker/temporal và UI; không gọi FPS
model là throughput hệ thống. Dùng cùng scope và GT/OP lock trước ablation test.

scripts/plot_far_delay.py --csv <parameter_sweep.csv> --output <new.html> tạo scatter
từ measured pairs; thiếu dữ liệu hiện annotation BLOCKED với0 points, không vẽ số giả.
Error-case CSV chỉ có header:0 verified cases, chưa đáp ứng ≥50. Mỗi case cần
video/sourceSHA/framePTS, GT review ID, prediction reference, verifier và cause:
FP/FN, reflection, lighting, red/orange object, steam/fog, blur, night, low-res smoke,
occlusion hoặc nguyên nhân thực quan sát khác. Không sinh50 hàng placeholder.

GVHD package: data/split/GT provenance, model+OP lock, A/B table, FAR–Delay chart,
ablation/day-night table, CPU/GPU/export evidence, ≥50 verified cases, code/env hashes,
limitations và tái lập commands. Chưa có các thực nghiệm nên chưa READY/PASS.
