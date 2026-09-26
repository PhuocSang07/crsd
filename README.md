# CSRD — Causal Step-Routing Distillation (cài đặt theo `CSRD_proposal_v3.pdf`)

Pilot hiện tại: **Qwen3-8B (thinking) → Qwen3-1.7B-Base**, dữ liệu s1K-Q8B, LoRA all-linear r = α = 64.
Cấu trúc code và convention theo `SpectralGuidedLearning/`: `src/` (Python, argparse + `--config` yaml),
`scripts/<phase>/<phase>_<track>.sh` (shell, `PROJECT_ENV`, `LOCAL_MODELS_ROOT`, `OPTS+=`, log vào `logs/`),
driver `project_commands_pilot.sh`, test CPU trong `tests/`.

> Trạng thái: code + test đã xong và chạy thông end-to-end trên CPU với model Qwen3 ngẫu nhiên tí hon
> (`scripts/smoke_test_pipeline.py`). **Chưa có kết quả thực nghiệm**: máy phát triển không có GPU, nên
> pilot 8B → 1.7B cần chạy trên node GPU (`bash project_commands_pilot.sh`).

## Ánh xạ proposal → code

| Proposal | File | Ghi chú |
|---|---|---|
| §6.2 s1K-Q8B: 8 trace/câu, T=0.6, top-p 0.95, top-k 20, 32k; lọc đúng + không bị cắt; math-verify / LLM-judge | `src/generate_traces.py` | `--stage generate` (vLLM, shard theo GPU) và `--stage select` (G0: ≥ 600 câu) |
| §6.2 tập dev (200) và held-out (300 trace đúng) | `generate_traces.py --source math-train`, `benchmarks.load_dev` | MATH train level 3–5, không trùng s1K/MATH500/AIME/AMC |
| §6.2 khử nhiễm 13-gram | `src/decontaminate.py` | |
| §4.1, Phụ lục B: nút trên span ký tự, tách `\n\n`, gộp < 40 ký tự, n ≤ 400; ánh xạ token qua offset | `src/step_nodes.py`, `src/data_prep.py` | A9: `--segment-mode paragraph/sentence/episode/chunk3`; A15: `--source s1k11-r1` |
| Phụ lục B: nhãn loại câu, câu neo | `src/anchor_labels.py` | heuristic (mặc định) hoặc `--labeler llm` |
| Định nghĩa 1: m_t(j), F(i), Z_t, R_t, trung bình theo hàng | `src/routing.py` | |
| §4.3 vertical score, kurtosis thô / excess kurtosis trừ nền, band [0.4,0.7) / [0.7,1.0], top-16/band, split-half | `src/receiver_heads.py`, `extract_routing.py --stage calibrate/select` | xuất cả `heads-{excess_bg,kurtosis,random,allband}.json` (A2/A5) |
| §4.3 target P⁽ᵇ⁾, Z⁽ᵇ⁾ (teacher offline) | `extract_routing.py --stage targets` | lưu `<id>.npz` (P, Z, rows, char_spans) |
| §4.4 target nhân quả (attention suppression, Eq. 3), J = top-24 | `src/causal_targets.py` | 20% trace train; 20 trace held-out cho D4 |
| §4.5 Eq. 4–7, lịch λ (0 trong 10% đầu, tăng tuyến tính 10% tiếp), chọn head student sau warmup CE | `src/csrd_trainer.py`, `src/train_sft.py` | `--csrd-lambda 0` = SFT (B1), cùng engine |
| §4.6 Bổ đề 1 | `tests/test_routing.py` | kiểm tra bằng số + gradcheck |
| §4.7 LoRA, CSRD-QK (adapter Q/K riêng, input detach), theo dõi cos(∇CE, ∇route) | `train_sft.attach_lora`, `csrd_trainer` | |
| §4.8 lấy mẫu m = 8 truy vấn/bước, softmax chính xác trên cả hàng | `csrd_trainer.sample_queries`, `routing.head_query_routing` | checkpoint theo head |
| §4.9 biến thể CSRD / -C / -A / -QK / -PQ | `scripts/train/train_q8b-1.7b.sh ARM` | + ablation `csrd-nomass` (A7), `csrd-band` (A5), `csrd-kurtosis` (A2), `csrd-causalonly` (A3), `csrd-b1/-b2` (A4); env `QUERIES` (A6), `LORA_R`/`QK_RANK` (A13), `D_MIN` (A1) |
| §5 D1–D5 + G1–G4 | `src/diagnostics.py` | D1: RG(b), Δμ(b), độ dốc theo log khoảng cách, bootstrap theo trace |
| §5 D6 QK-Restore | `src/qk_restore.py`, `extract_routing.py --qk-restore` | |
| §6.4 pass@1 / pass@3 không chệch, n = 16 (AIME/AMC), 4 (MATH500), bootstrap 2 tầng | `src/evaluate.py`, `src/pass_at_k.py` | B0: `--prompt-style zeroshot/fewshot` |
| §6.7, Phụ lục C error injection | `src/error_injection.py` | d ∈ {4, 16, 64}, 3 kiểu biến đổi, nhóm đối chứng |
| §7 pilot, cây quyết định | `project_commands_pilot.sh`, `src/pilot_report.py`, `src/compare_results.py` | G5 (pass@1 +1.5), G6 (overhead ≤ 25%) |

## Chạy

```bash
bash scripts/setup.sh                    # venv /mnt/local/uvenvs/crsd (PROJECT_ENV), cùng pin với SGL
GPUS="0 1 2 3 4 5 6 7" bash project_commands_pilot.sh
```

Từng bước (mỗi script tự bỏ qua phần đã có output):

```bash
bash scripts/gen/gen_s1k-q8b.sh            # data/q8b/s1k-traces.jsonl (+ .stats.json: G0)
bash scripts/gen/gen_heldout-q8b.sh        # data/q8b/heldout-traces.jsonl
bash scripts/data/data_q8b-1.7b.sh         # {s1k,heldout}-{teacher,student}.jsonl + nhãn câu neo
bash scripts/targets/targets_qwen3-8b.sh   # receiver heads, target P/Z, target nhân quả
bash scripts/train/train_q8b-1.7b.sh sft 42
LAMBDA=0.3 bash scripts/train/train_q8b-1.7b.sh csrd 42
bash scripts/diag/diag_q8b-1.7b.sh checkpoints/sft-q8b-1.7b-s42 sft-q8b-1.7b-s42   # D1-D6 -> G1, G2, G4 (+G3)
bash scripts/eval/eval_q8b-1.7b.sh checkpoints/csrd-l0.3-q8b-1.7b-s42 csrd-l0.3-q8b-1.7b-s42
bash scripts/inject/inject_q8b-1.7b.sh checkpoints/csrd-l0.3-q8b-1.7b-s42 csrd-l0.3-q8b-1.7b-s42
python src/compare_results.py --baseline sft-q8b-1.7b && python src/pilot_report.py
```

Kiểm thử (CPU, không cần GPU):

```bash
python -m pytest tests -q                                             # 38 test
python scripts/smoke_test_pipeline.py --tokenizer <thư mục tokenizer Qwen3>   # toàn pipeline trên model tí hon
```

Các test kiểm những mục trong checklist Phụ lục D: khối lượng attention tính theo khối khớp attention đầy đủ;
kiểm tra gradient bằng sai phân hữu hạn; kiểm tra Bổ đề 1 bằng số (kể cả khi student gán ~e⁻¹² attention cho nút
đích); CSRD-QK chỉ cập nhật A_QK bằng loss định tuyến và adapter chính chỉ nhận CE; ước lượng pass@k so với liệt kê
tổ hợp. Ngoài ra: q/k lấy từ hook khớp attention eager (~1e-7), suppression chỉ đổi các vị trí sau nút, và gradient
định tuyến đi qua gradient checkpointing tới LoRA q/k.

## Quyết định cài đặt (chỗ proposal chưa nói rõ, hoặc làm khác)

1. **Lấy q/k chính xác mà vẫn dùng SDPA/FlashAttention.** `attention_capture.install()` bọc hàm attention trong
   `AttentionInterface` của transformers; hook chỉ giữ tham chiếu tới `query` (sau q_norm + RoPE) và `key`, việc cắt
   head/vị trí làm sau forward. Làm bất kỳ phép tính nào bên trong hook sẽ lệch số tensor giữa forward và recompute
   của gradient checkpointing (lỗi này đã được tái hiện và có test).
2. **Attention của teacher**: thay vì đọc LSE từ FlashAttention (§4.8), tính lại softmax chính xác trên cả hàng từ q/k
   đã capture, theo khối truy vấn (không bao giờ tạo ma trận T×T). Kết quả chính xác như nhau và không phụ thuộc API
   của phiên bản flash-attn.
3. **Nút q** = nội dung user (instruction + câu hỏi). Token của template (`<|im_start|>…`), `<think>`/`</think>` và
   token dừng không thuộc nút nào, nên token attention-sink đầu chuỗi không lọt vào phân phối định tuyến.
   Nút đáp án a cũng là một hàng truy vấn.
4. **Band độ sâu** theo l/(L−1): teacher 36 tầng → B1 = 14–24, B2 = 25–35; student 28 tầng → B1 = 11–18, B2 = 19–27.
5. **Chọn head student**: điểm receiver của mọi head được tích luỹ từ các truy vấn đã lấy mẫu trong giai đoạn warmup
   chỉ CE (10% bước đầu), all-reduce giữa các rank, rồi cố định top-16/band (lưu `csrd-student-heads.json`, được giữ
   nguyên khi `--resume`).
6. **CE theo khối** (`masked_loss.chunked_cross_entropy`) thay cho fused linear CE của Liger: cùng giá trị/gradient
   (có test), không tạo logits 32k × 151k.
7. **DDP thay cho DeepSpeed ZeRO-3** với student 1.7B + LoRA: ZeRO-3 phân mảnh tham số nên CE theo khối không đọc
   được `lm_head.weight`, và hook gradient của CSRD-QK cần tham số đầy đủ. Có thể truyền `DS_CONFIG` (ZeRO-2).
8. **CSRD-QK**: thay vì hai lượt backward, một tensor hook trên tham số của A_QK *thay* gradient CE bằng gradient
   định tuyến (tính bằng `autograd.grad` từ q/k tính lại trên hidden state đã detach). DDP vẫn all-reduce giá trị đã
   thay. Khi lưu, cả hai adapter được merge vào trọng số đầy đủ để vLLM phục vụ đúng hàm đã huấn luyện;
   `adapters-separate/` giữ bản tách riêng.
9. **Target nhân quả**: KL của phân phối token kế tiếp dùng logits tại vị trí t−1 (phân phối sinh token t).
10. **Nhãn câu neo** mặc định là heuristic từ khoá (nhanh, cần báo cáo là xấp xỉ); `LABELER=llm` dùng Qwen3-8B như
    giao thức trong Phụ lục B (vẫn cần kiểm tra tay 300 bước, báo cáo accuracy và Cohen's κ).
11. **Prompt**: dạng khuyến nghị của Qwen3 (`{câu hỏi}\nPlease reason step by step, and put your final answer within \boxed{}.`),
    chat template thinking, render một lần bằng tokenizer teacher và dùng nguyên văn cho student, eval và chẩn đoán.
    Student Base học thêm token kết thúc lượt `<|im_end|>` (eos gốc của Base là `<|endoftext|>`). Khi sinh, vLLM dừng
    theo **token id** (`<|im_end|>`, `<|endoftext|>`), vì stop string không khớp được với token đặc biệt đã bị bỏ khỏi text.
12. **Độ dài sinh khi eval**: Qwen3-1.7B-Base có `max_position_embeddings = 32768`, nên `max_model_len = 32768` và
    `max_tokens = 31744` (dành 1024 token cho prompt), thay cho 32,768 token sinh như §6.4. Error injection giới hạn
    `max_tokens` theo phần context còn lại sau prefix và bỏ case còn < 4096 token.
13. **Dev và held-out** (MATH train level 3–5) loại mọi câu có 13-gram chung với s1K hoặc bốn tập test. s1K lấy đề từ
    chính MATH (`qfq/openaimath`); nếu không lọc thì 4/200 câu dev và 19/500 câu ứng viên held-out trùng s1K.
14. **Error injection**: mỗi case thuộc đúng một bucket theo khoảng cách tới lần dùng lại **đầu tiên**
    ([4,16), [16,64), [64,∞)); loại các giá trị r còn xuất hiện ở bước j+1..j+2 (vì chúng nằm trong prefix, student có thể
    chép lại); "phát hiện" được báo cáo dưới dạng hiệu so với tỉ lệ nêu r ở nhóm đối chứng (`detection_net`).
15. **Target nhân quả**: lượt chạy sạch đi qua cùng đường attention (mask tường minh, khối rỗng) với lượt bị chặn, nên
    KL chỉ phản ánh suppression chứ không lẫn sai khác giữa các kernel bf16; KL ở các vị trí trước nút j được lưu
    (`floor`) và phải xấp xỉ 0.
16. **Softmax định tuyến luôn ở fp32**, tắt autocast (hook thống kê head chạy bên trong forward, nơi accelerate bật
    autocast bf16). Model và vLLM chạy bf16; target lưu fp32; R theo từng head cho D4 lưu float16.
17. **G6** dùng tỉ số tổng `train_runtime_s` giữa CSRD và SFT (cùng dữ liệu, cùng số bước) làm đại diện cho "overhead ở 32k";
    `run-summary.json` ghi thêm `peak_memory_gb`. **D5** tính khoảng cách attention trung bình trên các token truy vấn của
    các hàng có |F(i)| ≥ 2, cho mọi head (`routing/mean-distance.npy`). **D6** giữ nguyên tập head của student, chỉ
    zero phần cập nhật q/k.
18. **λ trong pilot** là {0.3, 1} (Bảng 6); error injection và chẩn đoán cho CSRD mặc định chạy trên λ = 0.3. Ở bảng
    chính, λ phải được chọn trên tập dev như §6.3.

## Chưa làm (bước tiếp theo sau pilot)

- Baseline B2–B7 (token-level KL, Segment Selective SFT, SGL, P-ALIGN, RSR, MoLSAKI); trace ứng viên cho RSR đã được
  lưu trong trường `candidates`.
- Student 4B, teacher 32B, full fine-tuning (Giả thuyết 5); các script đã tách theo track nên thêm track mới chủ yếu
  là đổi `MODEL_NAME` và đường dẫn.
- A5 (trọng số head học được), A10 (RKD/CKA), A11 (kết hợp với baseline).
- G3, tiêu chí thứ hai (gap cục bộ ngay trước lỗi đầu tiên) cần gán nhãn bước lỗi đầu tiên; hiện chỉ có tiêu chí AUC.
- Error injection: "phát hiện" hiện chỉ so khớp số (chưa có LLM-judge theo rubric).
- Phân tích hành vi §6.7 ngoài error injection: anchor deletion, phân loại phản tư (xác nhận/sửa đổi), tương quan độ dài
  phản hồi với độ khó.
- Ablation A9 (cách phân đoạn): `data_q8b-1.7b.sh` đã sinh record theo `SEGMENT_MODE`, nhưng
  `targets_qwen3-8b.sh` và script train vẫn trỏ vào record mặc định (paragraph).
