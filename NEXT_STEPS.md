# CSRD — Cần làm gì tiếp theo

Lộ trình từ trạng thái hiện tại (code và test xong, chưa chạy GPU) đến bảng kết quả. Mọi lệnh chạy từ thư mục gốc repo;
tham số đầu của mọi script là `TRACK`. Chi tiết thiết kế: `README.md`.

## 0. Trạng thái (26/09/2026)

| Hạng mục | Trạng thái |
|---|---|
| Hai hướng × hai teacher: `read-q8b-1.7b`, `read-d32b-q8b` (chính), `gen-q8b-1.7b`, `gen-d32b-q8b` (ablation) | Code xong |
| Text student trùng từng byte/token với định dạng SGL của baseline (kiểm trên 30 mẫu s1K-1.1, cả Qwen3-8B và 1.7B-Base) | Đã kiểm |
| Hash nút khớp giữa student (SGL) và teacher Qwen3-8B / R1-Distill-32B (template riêng) trên 40 mẫu thật | Đã kiểm |
| File tín hiệu dùng lại được (`signals/*.safetensors`), dataset đọc từ file hoặc thư mục cho kết quả giống hệt | Đã kiểm |
| Unit test CPU | 42/42 pass |
| Smoke test toàn pipeline, teacher khác tokenizer (Qwen2 + tokenizer R1 → Qwen3) | Pass |
| Các shell script chạy nguyên văn trên model tí hon (data, tín hiệu teacher, train, chẩn đoán; DDP 2 tiến trình) | Pass |
| Chạy GPU thật | **Chưa** |

## 1. Chuẩn bị máy GPU

1. Clone `https://github.com/PhuocSang07/crsd` lên node GPU (8×H100 80 GB; teacher 32B cần 2 GPU cho mỗi tiến trình đọc).
2. Tải theo `download.txt`: Qwen3-8B, Qwen3-1.7B-Base, DeepSeek-R1-Distill-Qwen-32B; s1K-1.1, s1K, OpenR1-Math-220k,
   MATH-lighteval, AIME24/25, MATH-500, AMC. (Không có bản mirror thì script tự lấy từ HF Hub.)
3. `bash scripts/setup.sh`, rồi `python -m pytest tests -q` (phải ra 42 passed).
   Trên Lightning Studio: cài vào Python hệ thống như `SpectralGuidedLearning/scripts/lightning_run.sh`, tạo
   `$PROJECT_ENV/bin/activate` rỗng và symlink `$PROJECT_ENV/bin/python`, đặt `LOCAL_MODELS_ROOT`, `BENCH_DATA_ROOT=""`.
4. Đặt `GPUS="0 1 2 3 4 5 6 7"`.

## 2. Hướng chính — teacher đọc lại s1K-1.1 (`project_commands_read.sh`)

Làm `read-d32b-q8b` trước: student Qwen3-8B trùng với checkpoint P-ALIGN / SSFT bạn đã có, nên so sánh được ngay.

| # | Lệnh | Kiểm tra |
|---|---|---|
| 1 | `bash scripts/data/canonical.sh read-d32b-q8b` | 1000 trace s1K-1.1; 300 trace held-out OpenR1 |
| 2 | `bash scripts/data/records.sh read-d32b-q8b` | log `records-*`: không có `content_not_verbatim` / `empty_node` đáng kể; bin khoảng cách ≥ 64 có cặp |
| 3 | `bash scripts/targets/teacher_signals.sh read-d32b-q8b` | `data/teacher/d32b-s1k11/routing/selection-summary.json` (split-half, r ≈ .67 là mốc tham chiếu); không có `WARNING … KL before the suppressed node`; tạo ra `signals/d32b-s1k11-dmin4-excess_bg.safetensors` |
| 4 | `bash scripts/train/train.sh read-d32b-q8b sft 42` | loss giảm, `peak_memory_gb` trong `run-summary.json` |
| 5 | `bash scripts/eval/eval.sh read-d32b-q8b <ckpt P-ALIGN> palign-read-d32b-q8b-s42` (tương tự cho SSFT, SGL) | eval lại baseline có sẵn bằng cùng grader, cả hai protocol; số `palign` nên gần số bạn đã có |
| 6 | `bash scripts/diag/diag.sh read-d32b-q8b base base-read-d32b-q8b` rồi `… checkpoints/sft-read-d32b-q8b-s42 sft-read-d32b-q8b-s42` (sau eval với `DEV_ROLLOUTS=1`) | **G1, G2, G4** (`results/diag-…/diagnostics.md`), **G3** (`…-d3`), D6 (`…-qkrestore`) |
| 7 | `LAMBDA=0.3 bash scripts/train/train.sh read-d32b-q8b csrd 42` (thêm λ = 1, seed 43/44) | `logs/train-csrd-*.log`: `loss_route` giảm, `csrd_zS_b*` tiến về `csrd_zT_b*` |
| 8 | eval, error injection, `compare_results.py --track read-d32b-q8b`, `pilot_report.py --track read-d32b-q8b` | **G5, G6**, kiểm định hoán vị + Holm so với SFT và với baseline |

Cách đơn giản nhất: `BASELINES_read_d32b_q8b="palign:/path/palign ssft:/path/ssft sgl:/path/sgl" TRACKS=read-d32b-q8b bash project_commands_read.sh`.
Sau đó chạy `read-q8b-1.7b` (cặp pilot của proposal). Track này **chưa có baseline** trên Qwen3-1.7B-Base: train
SGL / P-ALIGN / SSFT bằng code gốc của chúng với cùng dữ liệu và cấu hình, rồi eval bằng `scripts/eval/eval.sh read-q8b-1.7b …`.

Theo dõi khi train CSRD:
- `grad_route_ratio` luôn > 1 → loss định tuyến lấn át CE: thử λ = 0.1.
- `grad_cos_ce_route` âm kéo dài → chạy arm `csrd-qk`.

**Quyết định** (§7): G1 fail → dừng CSRD, chuyển hướng dự phòng B; G1 đạt, G4 fail → `scripts/targets/causal_heads.sh TRACK`
rồi dùng arm `csrd-c`; G1 ∧ G3 ∧ G5 → chương trình đầy đủ.

## 3. Hướng ablation — teacher tự sinh dữ liệu (`project_commands_gen.sh`)

`bash scripts/gen/gen_traces.sh gen-q8b-1.7b` (và `gen-d32b-q8b`; chạy thử `LIMIT=20` trước, rồi xoá `data/gen/<teacher>`),
sau đó các pha giống mục 2. Gate **G0**: `data/gen/<teacher>/s1k-traces.jsonl.stats.json` phải có ≥ 600 câu. Câu hỏi của
hướng này: tín hiệu từ đúng tác giả trace có tốt hơn tín hiệu từ người đọc lại không (so với mục 2, cùng student).
Dữ liệu khác của baseline, nên muốn có baseline ở hướng này thì phải train lại chúng trên dữ liệu sinh.

## 4. Tái sử dụng tín hiệu 32B cho student khác

`signals/d32b-s1k11-dmin4-excess_bg.safetensors` không phụ thuộc student. Với một student mới (ví dụ Qwen3-1.7B-Base, hay Llama):
1. `python src/data_prep.py --canonical data/canonical/s1k11.jsonl --tokenizer <student> --style sgl --output-path <records>`
2. `python src/train_sft.py … --data-path <records> --signals signals/d32b-s1k11-dmin4-excess_bg.safetensors --csrd-lambda 0.3`

Không cần chạy lại teacher, miễn giữ nguyên cách phân đoạn (`paragraph`, 40 ký tự, ≤ 400 bước); dataset sẽ báo lỗi ngay nếu hash nút không khớp.
Thêm một track mới vào `scripts/common.sh` (ví dụ `read-d32b-1.7b`) là đủ để dùng toàn bộ script.

## 5. Sau pilot (nếu go) — theo Bảng 9

1. Bảng chính: 3 seed, n = 16 (bỏ `N_SAMPLES_MAP`), λ chọn trên dev; baseline B2 (token-level KL), B6 (RSR), B7 (MoLSAKI).
2. Ablation A1–A9, A12–A15 đã có arm/cờ (`scripts/train/train.sh` liệt kê); còn thiếu A5 (trọng số head học được), A10, A11.
3. Full fine-tuning; phân tích hành vi §6.7 còn lại; LLM-judge cho phát hiện lỗi; tiêu chí thứ hai của G3.
4. Trước khi nộp (Phụ lục E): phiên bản AMC12 của P-ALIGN (83 bài, `AI-MO/aimo-validation-amc`), giấy phép s1K / OpenR1 / Qwen3 /
   DeepSeek, quét lại công trình liên quan, kiểm tra tay 300 nhãn câu neo nếu dùng `LABELER=llm`.

## 6. Gửi lại để phân tích sau bước 6 của mục 2

- `data/teacher/d32b-s1k11/routing/selection-summary.json`
- `results/diag-sft-read-d32b-q8b-s42/diagnostics.md` và `…-d3/diagnostics.json`
- `results-palign/*/summary.json` của baseline eval lại (để đối chiếu với số bạn đã có)
- 20–30 dòng cuối của `logs/train-sft-read-d32b-q8b-s42.log`
