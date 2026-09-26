# CSRD — Cần làm gì tiếp theo

Tài liệu này là lộ trình từ trạng thái hiện tại (code + test xong, chưa chạy GPU) đến bảng kết quả chính.
Mọi lệnh chạy từ thư mục gốc repo. Chi tiết từng module: `README.md`.

## 0. Trạng thái hiện tại (26/09/2026)

| Hạng mục | Trạng thái |
|---|---|
| Pipeline pilot 8B → 1.7B (sinh trace → nút → target teacher → train SFT/CSRD → chẩn đoán → eval → error injection → gate) | Code xong |
| Unit test CPU (Bổ đề 1, gradcheck, blockwise vs attention đầy đủ, cô lập gradient CSRD-QK, pass@k…) | 38/38 pass |
| Smoke test toàn pipeline trên model Qwen3 ngẫu nhiên tí hon | Pass |
| DDP 2 tiến trình (CSRD và CSRD-QK) | Pass |
| Chạy nguyên văn các shell script targets / train (sft, csrd, csrd-qk) / diag trên model tí hon | Pass |
| Review độc lập đối chiếu proposal (3 lỗi lớn + 8 lỗi nhỏ: D6 thiếu pass@1, metric error injection có thể "chép" đáp án, nền nhiễu của target nhân quả, …) | Đã sửa hết |
| Sanity check trọng số thật Qwen3-0.6B → 0.6B-Base (CPU) | Chạy thông; có receiver head thật |
| Chạy trên GPU với Qwen3-8B → Qwen3-1.7B-Base | **Chưa** |

## 1. Chuẩn bị máy GPU (nửa ngày)

1. Clone repo lên node GPU (8×H100 như §7; tối thiểu 1 GPU 80 GB vẫn chạy được, chỉ chậm hơn).
2. Tải model/dataset theo `download.txt`: Qwen3-8B, Qwen3-1.7B-Base, s1K, MATH-lighteval, AIME24/25, MATH-500, AMC.
3. `bash scripts/setup.sh`, sau đó `python -m pytest tests -q` (phải ra 38 passed).
   - Trên Lightning Studio: làm như `SpectralGuidedLearning/scripts/lightning_run.sh` (cài vào Python hệ thống, tạo
     `$PROJECT_ENV/bin/activate` rỗng, symlink `$PROJECT_ENV/bin/python`), đặt `LOCAL_MODELS_ROOT`, `BENCH_DATA_ROOT=""`,
     và `DATASET_NAME=<id HF>` cho hai script sinh trace.
4. Đặt `GPUS="0 1 2 3 4 5 6 7"`.

## 2. Chạy thử nhỏ (1–2 giờ)

```bash
LIMIT=20 bash scripts/gen/gen_s1k-q8b.sh
cat data/q8b/s1k-traces.jsonl.stats.json          # response có <think>…</think>? tỉ lệ đúng hợp lý?
rm -rf data/q8b/s1k-raw* data/q8b/s1k-traces.jsonl*   # bắt buộc, nếu không lần chạy thật sẽ bỏ qua các shard cũ
```

## 3. Pilot tuần 1 — dữ liệu và chẩn đoán

| # | Lệnh | Kiểm tra / gate |
|---|---|---|
| 1 | `bash scripts/gen/gen_s1k-q8b.sh` | **G0**: `kept_questions ≥ 600`. Nếu không đạt: bổ sung đề độ khó vừa (§6.2) |
| 2 | `bash scripts/gen/gen_heldout-q8b.sh` | khoảng 300 trace held-out |
| 3 | `bash scripts/data/data_q8b-1.7b.sh` | phân bố số bước; bin khoảng cách ≥ 64 phải có cặp; `decontamination.json` sạch |
| 4 | `bash scripts/targets/targets_qwen3-8b.sh` | `data/q8b/routing-teacher/selection-summary.json`: split-half (tham chiếu r ≈ .67) và độ trùng giữa hai cách chọn head; `WARNING … KL before the suppressed node` trong log nhân quả phải không xuất hiện |
| 5 | `bash scripts/train/train_q8b-1.7b.sh sft 42` | loss giảm, không OOM; `peak_memory_gb` trong `run-summary.json` |
| 6 | `bash scripts/diag/diag_q8b-1.7b.sh base base` | mốc trước huấn luyện |
| 7 | `DEV_ROLLOUTS=1 N_SAMPLES_MAP=aime24=8,aime25=8,amc12=8 bash scripts/eval/eval_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42` | pass@1 của SFT; truncation rate |
| 8 | `bash scripts/diag/diag_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42` | **G1, G2, G4** (`results/diag-…/diagnostics.md`), **G3** (`…-d3/`), RG sau QK-Restore (`…-qkrestore/`) |
| 9 | `python src/qk_restore.py --adapter checkpoints/sft-q8b-1.7b-s42 --output-dir checkpoints/sft-qkrestore-q8b-1.7b-s42`, rồi eval như bước 7 | **D6**: pass@1 của SFT gần như không đổi sau QK-Restore? |

**Quyết định sau tuần 1** (§7):
- G1 fail → dừng CSRD, chuyển sang hướng dự phòng B (bài phân tích: routing gap không xuất hiện ở Transformer thuần softmax).
- G1 đạt, G4 fail → `bash scripts/targets/causal_heads_qwen3-8b.sh`, tuần 2 dùng arm `csrd-c`.
- G1 đạt → sang tuần 2.

## 4. Pilot tuần 2 — can thiệp

Chạy phần "WEEK 2" trong `project_commands_pilot.sh`: SFT và CSRD với λ ∈ {0.3, 1} × seed {42, 43, 44}, eval
(n = 8 cho AIME/AMC, 4 cho MATH500), B0 zero-shot và few-shot, error injection khoảng 100 case, chẩn đoán cho CSRD, rồi:

```bash
python src/compare_results.py --baseline sft-q8b-1.7b   # bảng mean ± std, kiểm định hoán vị theo cặp + Holm, G5
python src/pilot_report.py                              # G0–G6 và nhánh quyết định
```

Theo dõi trong `logs/train-csrd-*.log`:
- `grad_route_ratio` luôn > 1 → loss định tuyến lấn át CE: thêm λ = 0.1.
- `grad_cos_ce_route` âm kéo dài → chạy arm `csrd-qk` (§4.7).
- `csrd_zS_b*` tiến dần về `csrd_zT_b*` → L_mass đang có tác dụng.

**Quyết định sau tuần 2:** G1 ∧ G3 ∧ G5 → chương trình đầy đủ (mục 5); G1 đạt nhưng G3 và G5 fail → hướng dự phòng A (§11.2).

## 5. Sau pilot (nếu go) — theo lịch T2–T4 (Bảng 9)

1. **Baseline (T2)**, cùng dữ liệu s1K-Q8B và cùng cấu hình LoRA:
   - B2 token-level KL: lưu offline logit top-k của Qwen3-8B (cùng tokenizer).
   - B3 Segment Selective SFT: tính Integrated Gradients cho khoảng 1k trace (mã gốc SiyuanWangw/SegmentSelectiveSFT).
   - B4 SGL: tái cài đặt, có thể dùng lại `SpectralGuidedLearning/` (ghi rõ là bản tái cài đặt).
   - B5 P-ALIGN (NEUIR/P-ALIGN), B6 RSR (trường `candidates` đã lưu 7 trace ứng viên mỗi câu),
     B7 MoLSAKI (dùng lại pipeline trích attention của CSRD).
2. **Bảng chính (T3)**: thêm track Qwen3-4B-Base (sao chép `*_q8b-1.7b.sh` thành `*_q8b-4b.sh`, đổi `MODEL_NAME`),
   3 seed, n = 16 cho AIME/AMC.
3. **Ablation**: A1–A9 và A12–A15 đã có cờ/arm; còn thiếu A5 (trọng số head học được), A10 (RKD/CKA), A11 (kết hợp với
   baseline). A9 cần nối `SEGMENT_MODE` vào script targets và train.
4. **Mở rộng (T4)**: teacher Qwen3-32B → 4B-Base, full fine-tuning 4B (lr 1e-5), biến thể trace R1
   (`data_prep.py --source s1k11-r1`, A15).
5. **Phân tích hành vi (§6.7)** chưa có code: anchor deletion, phân loại phản tư xác nhận/sửa đổi, tương quan độ dài
   phản hồi với độ khó; LLM-judge theo rubric cho phát hiện lỗi trong error injection; tiêu chí thứ hai của G3 (gap cục bộ
   ngay trước lỗi đầu tiên, cần gán nhãn bước lỗi).
6. **Trước khi nộp** (Phụ lục E): xác nhận phiên bản AMC12 của P-ALIGN, giấy phép s1K và Qwen3, quét lại công trình
   liên quan, kiểm tra tay 300 nhãn câu neo nếu dùng `LABELER=llm` (báo cáo accuracy và Cohen's κ).

## 6. Gửi lại để phân tích sau tuần 1

- `data/q8b/s1k-traces.jsonl.stats.json`
- `logs/select-heads-q8b.log`
- `results/diag-sft-q8b-1.7b-s42/diagnostics.md` và `…-d3/diagnostics.json`
- 20–30 dòng cuối của `logs/train-sft-q8b-1.7b-s42.log`
