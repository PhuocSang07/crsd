# CSRD — Causal Step-Routing Distillation (cài đặt theo `CSRD_proposal_v3.pdf`)

Cấu trúc code theo `SpectralGuidedLearning/`: `src/` (Python, argparse + `--config` yaml), `scripts/<pha>/*.sh`
(shell, `PROJECT_ENV`, `LOCAL_MODELS_ROOT`, `OPTS+=`, log vào `logs/`), test CPU trong `tests/`.

> Trạng thái: code và test đã xong, toàn pipeline chạy thông trên CPU với model tí hon (kể cả teacher khác tokenizer).
> **Chưa có kết quả thực nghiệm GPU.** Lộ trình chạy: `NEXT_STEPS.md`.

## Thiết kế thực nghiệm

Mọi arm (SFT, CSRD, và các baseline SGL / P-ALIGN / SSFT) train trên **cùng một text**, cùng student, cùng
cấu hình LoRA/lịch/batch, và được eval bằng **cùng một script, một grader, hai protocol**. CSRD chỉ khác ở tín hiệu
định tuyến mà teacher đọc được trên chính dữ liệu đó.

| Track | Teacher | Dữ liệu train (chung cho mọi arm) | Student | Vai trò |
|---|---|---|---|---|
| `read-q8b-r1.5b` | Qwen3-8B đọc lại | s1K-1.1 (trace DeepSeek-R1) | DeepSeek-R1-Distill-Qwen-1.5B | **chính cho cặp nhỏ** (cùng student với baseline SGL) |
| `read-q8b-1.7b` | Qwen3-8B đọc lại | s1K-1.1 (trace DeepSeek-R1) | Qwen3-1.7B-Base | bỏ: với cấu hình SGL, student Base không học được format R1 |
| `read-d32b-q8b` | DeepSeek-R1-Distill-Qwen-32B đọc lại | s1K-1.1 | Qwen3-8B | chính (cùng student với P-ALIGN/SSFT) |
| `gen-q8b-1.7b` | Qwen3-8B tự viết rồi đọc | s1K-Q8B | Qwen3-1.7B-Base | ablation: teacher = tác giả (§6.1) |
| `gen-d32b-q8b` | R1-Distill-32B tự viết rồi đọc | s1K-D32B | Qwen3-8B | ablation |

- **Định dạng text của student = định dạng SGL**, trùng từng byte và từng token với `SpectralGuidedLearning/src/data_prep.py`
  (đã kiểm trên s1K-1.1 với tokenizer Qwen3-8B và Qwen3-1.7B-Base): chat template với `enable_thinking=False`, user turn
  `Please reason step by step, and put your final answer within \boxed{}.{problem}`, response `{trajectory}\n\n\n{attempt}`,
  thêm eos của tokenizer.
- **Teacher đọc trace như lập luận của chính nó**: template thinking của nó (`<think>…</think>`; R1-Distill tự mở `<think>`).
- **Thông số train** theo `SpectralGuidedLearning/project_commands_spectral_r1-qwen-1.5b.sh`: LoRA all-linear r = α = 16,
  dropout 0.05, lr 5e-5 cosine xuống 1e-5, warmup 0.1, 3 epoch, batch hiệu dụng 32, seed 42, 32k token, DeepSpeed ZeRO-2 offload.
- **Eval**: `proposal` (§6.4: T 0.6, top-p 0.95, top-k 20, n = 16 AIME/AMC và 4 MATH500, context 32k, pass@k không chệch) và
  `palign` (cách eval của baseline: T 0.6, top-p 0.9, repetition penalty 1.05, n = 3, context 4096). Grader chung cho mọi arm:
  math_verify HOẶC oat_math_grader (như SGL). Checkpoint baseline có sẵn được eval lại bằng đúng script này.

## Tín hiệu teacher dùng lại được

Nội dung trace được lưu một lần (`data/canonical/*.jsonl`: câu hỏi, thinking, đáp án), rồi render riêng cho từng model.
Các nút (câu hỏi, các bước, đáp án) được cắt từ **nội dung**, nên mọi cách render cho cùng một tập nút; mỗi nút mang
hash của text. Tín hiệu teacher trên một bộ dữ liệu được đóng gói thành **một file**
`signals/<teacher>-<data>-dmin<d>-<score>.safetensors` (P, Z, rows, hash nút, target nhân quả; metadata ghi teacher,
dữ liệu, style, cách phân đoạn, d_min, danh sách head). Mọi student sau này, kể cả khác tokenizer, train trực tiếp từ file đó:
chỉ cần record của student (`data_prep.py --style sgl`) có cùng hash nút. Ví dụ, tín hiệu của 32B trên s1K-1.1 tính một lần,
dùng được cho Qwen3-8B, Qwen3-1.7B-Base hay Llama.

## Ánh xạ proposal → code

| Proposal | File | Ghi chú |
|---|---|---|
| Nội dung chuẩn và cách render cho từng model | `src/prompting.py`, `src/build_canonical.py` | nguồn: s1K-1.1, OpenR1-Math (held-out), trace tự sinh, rollout |
| §6.2 sinh trace teacher (hướng gen), lọc đúng, LLM-judge | `src/generate_traces.py` | vLLM, tensor parallel cho 32B; judge Qwen3-8B |
| §6.2 dev (200) và held-out | `benchmarks.load_dev`, `build_canonical --source openr1` | 13-gram không trùng s1K và 4 tập test |
| §4.1, Phụ lục B: nút theo nội dung, tách `\n\n`, gộp < 40 ký tự, n ≤ 400 | `src/step_nodes.py`, `src/data_prep.py` | A9: `SEGMENT_MODE` |
| Phụ lục B: nhãn loại câu, câu neo | `src/anchor_labels.py` | heuristic (mặc định) hoặc LLM |
| Định nghĩa 1, Eq. 4–7, Bổ đề 1 | `src/routing.py` | |
| §4.3 receiver head, band, top-16/band, split-half | `src/receiver_heads.py`, `src/extract_routing.py` | teacher lớn chia trên nhiều GPU (`--device-map auto`) |
| §4.4 target nhân quả (attention suppression) | `src/causal_targets.py` | lượt sạch và lượt chặn đi cùng đường kernel |
| File tín hiệu dùng lại được | `src/signal_bank.py` | `pack` / `info`; `SignalSource` đọc cả file lẫn thư mục |
| §4.5–4.9 hàm mục tiêu, lịch λ, chọn head student, biến thể CSRD / -A / -C / -PQ / -QK | `src/csrd_trainer.py`, `src/train_sft.py` | `--csrd-lambda 0` = SFT, cùng engine |
| §5 D1–D6, gate G1–G4 | `src/diagnostics.py`, `src/qk_restore.py` | |
| §6.4 eval, pass@k, bootstrap, kiểm định hoán vị + Holm | `src/evaluate.py`, `src/pass_at_k.py`, `src/compare_results.py` | 2 protocol, grader `palign_grader.py` |
| §6.7, Phụ lục C error injection | `src/error_injection.py` | bucket theo lần dùng lại đầu tiên, trừ nền đối chứng |
| §7 gate G0–G6 và cây quyết định | `src/pilot_report.py` | theo track |

## Chạy

```bash
bash scripts/setup.sh                         # venv /mnt/local/uvenvs/crsd (PROJECT_ENV)
GPUS="0 1 2 3 4 5 6 7" bash project_commands_read.sh   # hướng chính: đọc lại s1K-1.1
GPUS="0 1 2 3 4 5 6 7" bash project_commands_gen.sh    # ablation: teacher tự sinh dữ liệu
```

Từng pha (tham số đầu luôn là `TRACK`; mỗi pha tự bỏ qua phần đã có output):

```bash
bash scripts/gen/gen_traces.sh gen-q8b-1.7b          # chỉ cho track gen-*
bash scripts/data/canonical.sh read-d32b-q8b
bash scripts/data/records.sh read-d32b-q8b
bash scripts/targets/teacher_signals.sh read-d32b-q8b   # -> signals/d32b-s1k11-dmin4-excess_bg.safetensors
bash scripts/train/train.sh read-d32b-q8b sft 42
LAMBDA=0.3 bash scripts/train/train.sh read-d32b-q8b csrd 42
bash scripts/eval/eval.sh read-d32b-q8b checkpoints/csrd-l0.3-read-d32b-q8b-s42 csrd-l0.3-read-d32b-q8b-s42
bash scripts/eval/eval.sh read-d32b-q8b /path/to/palign-qwen3-8b palign-read-d32b-q8b-s42   # baseline có sẵn
bash scripts/diag/diag.sh read-d32b-q8b checkpoints/sft-read-d32b-q8b-s42 sft-read-d32b-q8b-s42
python src/compare_results.py --results-dir results-proposal --track read-d32b-q8b
python src/pilot_report.py --track read-d32b-q8b
```

### Script theo style SGL: R1-Distill-Qwen-1.5B, read mode (B200 / H200)

Mỗi pha một script riêng cho model, biến viết hoa, `OPTS+=`, `CMD=…; echo; ${CMD} | tee logs/…`, driver
`project_commands_*.sh` chạy idempotent — đúng khuôn `SpectralGuidedLearning/scripts/spectral/spectral_lora_r1-qwen-1.5b.sh`.
Pipeline chạy offline (`HF_HUB_OFFLINE=1`, tắt usage stats của vLLM); model/data phải có sẵn ở `LOCAL_MODELS_ROOT` /
`LOCAL_DATA_ROOT` (danh sách: `download.txt`).

Server mới (ví dụ H200; cần driver NVIDIA ≥ 580 cho wheel cu130):

```bash
export PROJECT_ENV=/path/to/uvenvs/crsd LOCAL_MODELS_ROOT=/path/to/models LOCAL_DATA_ROOT=/path/to/datasets
bash scripts/setup.sh                                  # venv (uv, cần PyPI), pin = crsd.txt
bash scripts/data/download_r1-qwen-1.5b.sh             # Phase 0 (cần HF Hub): s1K-1.1, 4 tập test, Qwen3-8B, R1-Distill-1.5B
GPUS="0" bash project_commands_csrd_r1-qwen-1.5b.sh   # data -> bank -> mỗi biến thể: train CSRD -> eval 4k -> compare
```

Biến thể (`VARIANTS` = `tên:λ:mass_ratio:bands:d_min`, chạy lần lượt, biến thể đã xong được bỏ qua). Mặc định, theo
phân tích λ = 0.3 trong `EXPERIMENT_LOG.md` (far mass Z vượt teacher 2.4×): `main` (λ 0.1), `mass1` (mass_ratio 1.0),
`band1` (chỉ band giữa), `band1-mass1`. `d_min ≠ 4` tự dựng thêm bank teacher tương ứng (head dùng chung).
Script train cũng nhận trực tiếp `CSRD_LAMBDA`, `CSRD_MASS_RATIO`, `CSRD_BANDS`, `CSRD_D_MIN`, `DS_CONFIG=""` (bỏ DeepSpeed).

Từng pha:

```bash
GPUS="0" bash project_commands_csrd_r1-qwen-1.5b.sh     # data -> teacher bank -> (train -> eval) cho từng biến thể -> compare
VARIANTS="main:0.1:0.1:0,1:4 dmin16:0.1:0.1:0,1:16" GPUS="0" bash project_commands_csrd_r1-qwen-1.5b.sh   # tự chọn biến thể
bash scripts/data/data_r1-qwen-1.5b.sh            # Phase 1: canonical s1K-1.1 + record teacher (Qwen3-8B) / student (SGL format)
GPUS="0 1 2 3" bash scripts/teacher/teacher_qwen3-8b.sh   # Phase 2: bank tín hiệu Qwen3-8B (bỏ qua nếu đã có file bank)
GPUS="0 1 2 3" CSRD_LAMBDA=0.1 bash scripts/csrd/csrd_lora_r1-qwen-1.5b.sh   # Phase 3: -> checkpoints/csrd-lora-l0.1-r1-qwen-1.5b
bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/csrd-lora-l0.1-r1-qwen-1.5b csrd-lora-l0.1-r1-qwen-1.5b   # Phase 4: P-ALIGN
bash scripts/eval/eval32k_r1-qwen-1.5b.sh        # cap 32k, MATH500 + AIME24, mọi checkpoints/*-r1-qwen-1.5b -> results-32k/
```

Train: `train_sft.py` + DeepSpeed ZeRO-2 offload, LoRA r16/α16, lr 5e-5 → 1e-5, 3 epoch, batch hiệu dụng 32 (= #GPU × GA),
seed 42, 32k token — y hệt arm spectral-lora của SGL, chỉ thêm khối `--csrd-*`. Bank tín hiệu không phụ thuộc student:
chép `signals/q8b-s1k11-dmin4-excess_bg.safetensors` sang máy mới để bỏ qua Phase 2. SFT đối chứng = arm vanilla của SGL
(chép `results/vanilla-r1-qwen-1.5b` của SGL vào `results-palign/` để `compare_results.py` kiểm định cặp).

Kiểm thử (CPU): `python -m pytest tests -q` và
`python scripts/smoke_test_pipeline.py --teacher-tokenizer <tokenizer R1-Distill> --student-tokenizer <tokenizer Qwen3>`.

## Quyết định cài đặt (chỗ proposal chưa nói rõ, hoặc làm khác)

1. **Lấy q/k chính xác mà vẫn dùng SDPA/FlashAttention**: bọc hàm attention trong `AttentionInterface` của transformers; hook
   chỉ giữ tham chiếu tới `query`/`key` và cắt head/vị trí sau forward (một phép tính bên trong hook sẽ làm lệch
   gradient checkpointing — đã tái hiện và có test). Chạy được với Qwen3 (có q_norm/k_norm) và Qwen2/R1-Distill.
2. **Attention của teacher**: tính lại softmax chính xác trên cả hàng từ q/k đã capture, theo khối truy vấn, thay vì đọc LSE
   từ FlashAttention (§4.8). Kết quả chính xác như nhau, không phụ thuộc API flash-attn.
3. **Nút q** = nội dung user; token template, `<think>`/`</think>` và token dừng không thuộc nút nào. Nút đáp án cũng là một hàng.
4. **Band độ sâu** theo l/(L−1): Qwen3-8B (36 tầng) B1 = 14–24, B2 = 25–35; R1-Distill-32B (64 tầng) B1 = 26–44, B2 = 45–63;
   student 28 tầng B1 = 11–18, B2 = 19–27.
5. **Chọn head student**: tích luỹ điểm receiver của mọi head từ truy vấn đã lấy mẫu trong 10% bước đầu (chỉ CE), all-reduce
   giữa các rank, cố định top-16/band (lưu `csrd-student-heads.json`, giữ nguyên khi `--resume`).
6. **CE theo khối** thay cho fused linear CE của Liger (cùng giá trị/gradient, có test).
7. **CSRD-QK**: tensor hook thay gradient CE trên A_QK bằng gradient định tuyến (chia cho số bước tích luỹ như accelerate làm
   với loss); cần DDP, không chạy với DeepSpeed. Khi lưu, merge cả hai adapter thành checkpoint đầy đủ; `adapters-separate/` giữ bản tách.
8. **Target nhân quả**: KL của phân phối token kế tiếp tại vị trí t−1; lượt sạch đi qua cùng đường attention (mask tường minh, khối rỗng)
   với lượt bị chặn; KL ở các vị trí trước nút j được lưu (`floor`) và phải xấp xỉ 0.
9. **Softmax định tuyến luôn fp32**, tắt autocast (hook thống kê head chạy trong forward, nơi accelerate bật autocast bf16).
   Model và vLLM chạy bf16; target lưu fp32; R theo từng head cho D4 lưu float16.
10. **Prompt và định dạng**: theo SGL (xem trên) cho student ở mọi track, kể cả khi eval; prompt sinh trace của teacher dùng cùng user turn.
    vLLM dừng theo **token id** (`<|im_end|>`, `<|endoftext|>`, `<｜end▁of▁sentence｜>`), vì stop string không khớp token đặc biệt.
11. **Context eval 32k**: Qwen3-*-Base có `max_position_embeddings = 32768` nên `max_tokens = 31744` ở protocol proposal.
12. **Held-out của hướng đọc lại**: 300 trace R1 đúng và hoàn chỉnh từ OpenR1-Math-220k (các baseline đã dùng hết 1000 câu s1K-1.1).
    Dev và held-out MATH-train loại mọi câu có 13-gram chung với s1K hoặc 4 tập test.
13. **Error injection**: case dựng từ record của student (định dạng SGL); bucket theo khoảng cách tới lần dùng lại đầu tiên; loại giá trị
    còn xuất hiện ở bước j+1..j+2; "phát hiện" báo cáo dạng hiệu so với nhóm đối chứng.
14. **G6** dùng tỉ số `train_runtime_s` CSRD/SFT; `run-summary.json` ghi `peak_memory_gb`. **D5** so khoảng cách attention của mọi
    head (`routing/mean-distance.npy`). **D6** giữ nguyên tập head, chỉ zero cập nhật q/k.
15. **λ trong pilot** là {0.3, 1}; error injection và chẩn đoán CSRD mặc định trên λ = 0.3. Bảng chính phải chọn λ trên dev (§6.3).
16. **Thứ tự mẫu**: record giữ thứ tự của s1K-1.1, còn SGL đọc theo luồng đã xáo (seed 42). Tập text giống hệt; Trainer vẫn tự xáo
    mỗi epoch, nên chỉ khác thứ tự batch — chỉ quan trọng nếu muốn tái lập từng bước lần chạy "vanilla" của SGL.
17. **Mang file tín hiệu sang máy khác**: chỉ cần file `.safetensors` (nó mang theo danh sách head của teacher);
    `scripts/targets/teacher_signals.sh` tự dựng lại `heads-*.json` từ file (`signal_bank.py heads`). Riêng tín hiệu held-out cho
    chẩn đoán vẫn cần teacher.

## Chưa làm

- Baseline cho student Qwen3-1.7B-Base (track `read-q8b-1.7b`): cần train SGL / P-ALIGN / SSFT bằng code gốc của chúng trên 1.7B;
  baseline B2 (token-level KL), B6 (RSR), B7 (MoLSAKI).
- Full fine-tuning; A5 (trọng số head học được), A10 (RKD/CKA), A11 (kết hợp với baseline).
- G3 tiêu chí thứ hai (gap cục bộ trước lỗi đầu tiên); LLM-judge cho phát hiện lỗi; phân tích hành vi §6.7 khác (anchor deletion,
  phân loại phản tư, độ dài theo độ khó).
- SSFT được train với prompt "câu hỏi trước" và chế độ thinking mặc định, khác định dạng SGL: khi eval chung bằng prompt SGL,
  cần ghi chú điều này (hoặc eval thêm với `PROMPT_STYLE=thinking`).
