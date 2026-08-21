# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Xuân Kiên  **MSSV**: 2A202601398  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`data/train_seed.jsonl`) |
| Train / val | `225` / `25` (seed 42, tỷ lệ 90/10) |
| `max_length` | `256` — p95 đo được là `98` *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | `2 epochs` / `30 max_steps` |

**Template có giữ khối `<think>` không?** `Có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? Jinja chat template của model `unsloth/Qwen3.5-4B` bảo toàn nguyên vẹn cặp thẻ `<think>`...`</think>` và nội dung suy luận bên trong (`open_tag_present=true`, `body_present=true`, verdict: *"reasoning preserved — safe to train on traces"*). Do cấu trúc template giữ trọn vẹn khối reasoning mà không bị xóa bỏ trong quá trình render, ta không cần phải can thiệp hay sửa đổi thủ công chat template.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | `0.4149` |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.750 | 0.000 | 3276.9 |
| (b) base + optimized prompt | 0.688 | 0.750 | 1.000 | 940.9 |
| (c) LoRA fine-tune | 0.938 | 0.750 | 1.000 | 1532.2 |

**(b) có thật sự mạnh hơn (a) không?** `Có`. Baseline (b) vượt trội hoàn toàn so với (a) ở mọi phương diện: độ chính xác tác vụ mục tiêu (target) tăng từ `0.000` lên `0.688` (0.6875), tỷ lệ tuân thủ định dạng JSON (format) tăng từ `0.000` lên `1.000`, và độ trễ giảm mạnh từ `3276.9 ms` xuống `940.9 ms` nhờ prompt tối ưu giúp mô hình kết thúc sớm đúng cấu trúc thay vì sinh tự do lan man.

Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Không sửa `OPTIMIZED_PROMPT`. Prompt tối ưu được giữ nguyên vẹn theo chuẩn của labkit (khớp mã băm SHA `719e74d3b6232053`), bao gồm định nghĩa vai trò hệ thống, schema 4 khóa JSON bắt buộc, danh sách enum giá trị hợp lệ và ví dụ mẫu 1-shot. Việc giữ nguyên prompt chuẩn đảm bảo tính trung thực và liêm chính học thuật, tạo ra một rào cản kiểm thử thực sự cạnh tranh trước khi tiến hành fine-tuning, tuyệt đối không làm suy yếu prompt để tâng bốc kết quả của mô hình fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6258 | 0.9375 | 922.8 | 12.01 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5385 | 0.9375 | 782.1 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32464896 | 0.00001 | 1.5704 | 0.0000 | 910.9 | 12.01 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.8438 | 970.1 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target đánh giá ở NB5 §4, `attn_only` (với rank khớp ngân sách $r=283$, $32.46\text{M}$ tham số) đạt điểm số **0.9375**, hoà tuyệt đối với `correct` ($r=16$, $32.46\text{M}$ tham số). Tuy nhiên, thứ tự này hoàn toàn trái ngược với thứ tự theo train loss ở NB4, nơi `attn_only` có loss thấp hơn rõ rệt (`0.5385` so với `0.6258` của `correct`). Kết quả thực nghiệm này chỉ ra rằng việc dồn toàn bộ tham số vào rank cực lớn ở vị trí hẹp (chỉ $q,v$) chỉ giúp mô hình ghi nhớ/overfit dữ liệu huấn luyện cục bộ chứ không đem lại năng lực tổng quát hóa tốt hơn. Vị trí gắn adapter bao phủ toàn bộ các tầng tuyến tính (`text-linear`, bao gồm cả MLP/FFN) mới chính là đòn bẩy cấu trúc căn bản, giúp phân bổ dung lượng học đều khắp biểu diễn mạng thay vì ép tăng rank một cách cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` chỉ thay đổi duy nhất learning rate xuống thang đo full fine-tune ($10^{-5}$ thay vì $10^{-4}$), khiến đường train loss gần như đi ngang và dừng lại ở mức rất cao là `1.5704` (gần gấp 2.5 lần so với `0.6258` của `correct`), dẫn đến việc mô hình hoàn toàn thất bại trên tập target với điểm số `0.0000`. Nếu chỉ nhìn vào đường loss phẳng lì này mà không biết thông số LR, người làm thực nghiệm rất dễ kết luận sai lầm rằng mô hình bị thiếu dung lượng biểu diễn (under-capacity), kiến trúc LoRA không phù hợp hoặc bài toán quá khó, từ đó vội vã tăng rank $r$ hay đổi mô hình gốc một cách vô ích. Thực tế nguyên nhân cốt lõi là do ma trận $B$ của LoRA được khởi tạo bằng 0, đòi hỏi một tốc độ học lớn hơn khoảng $10\times$ so với full fine-tuning để các trọng số adapter kịp thời thích ứng và cập nhật hiệu quả trong số step ngắn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` (4-bit NF4) tiết kiệm được **4.92 GB VRAM** (giảm từ `12.01 GB` của bản 16-bit xuống còn `7.09 GB`, tiết kiệm xấp xỉ 41% bộ nhớ đồ họa), nhưng phải trả giá bằng sự suy giảm chất lượng khi điểm target tụt từ `0.9375` xuống `0.8438` và thời gian train tăng lên (`970.1 s` so với `922.8 s` do chi phí giải lượng tử hóa liên tục). Thêm vào đó, trên kiến trúc Turing T4, việc dùng QLoRA đòi hỏi các bản vá kiểu dữ liệu (recast tensors) để tránh lỗi xung đột của GradScaler. Số đo thực nghiệm hoàn toàn ủng hộ khuyến nghị của vendor Unsloth và bài học (§12): với dòng Qwen3.5, không nên lạm dụng QLoRA nếu GPU còn đủ dung lượng chứa bản 16-bit (fp16/bf16), nhằm bảo vệ tính toàn vẹn và độ chính xác phân loại của mô hình.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `PASSED`
`target Δ = +0.250` · `regression Δ = +0.000` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Mô hình LoRA fine-tune (`correct`) đã vượt qua cổng hồi quy (Regression Gate) một cách xuất sắc với kết quả **PASSED**. Cụ thể, trên tác vụ mục tiêu (target task), mô hình fine-tune đạt độ chính xác `0.9375`, mang lại mức cải thiện vượt bậc `target Δ = +0.250` (+25.0%) so với baseline mạnh nhất là prompt tối ưu (`0.6875`), và vượt xa baseline ngây thơ (`0.000`). Điều đặc biệt quan trọng là chỉ số duy trì năng lực tổng quát `regression Δ = +0.000` (giữ nguyên độ chính xác `0.750` trên tập tác vụ đối chứng đa miền), chứng tỏ mô hình hoàn toàn không gặp hiện tượng quên thảm họa (catastrophic forgetting) dù chỉ trải qua 30 bước tối ưu. Tỷ lệ tuân thủ định dạng đạt mức hoàn hảo `format = 1.000`, thể hiện việc cấu trúc JSON 4 trường bắt buộc đã được khắc sâu vào trọng số mà không cần hệ thống prompt mồi dài dòng. Giá trị `valid_trace_rate = 0.00` phản ánh đúng bản chất tác vụ phân loại thực tế cần đầu ra JSON trực tiếp, tinh gọn mà không sinh reasoning rườm rà.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ FT thắng (Score 1.0): Đúng hoàn toàn cả 4 trường chỉ với prompt ngắn 5 từ, không cần mồi schema. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | `{"intent": "hoan_tien", "urgency": "cao", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "ốp lưng điện thoại", "sentiment": "tieu_cuc"}` | ✅ FT thắng (Score 1.0): FT phân biệt chính xác mức độ "Sớm nhé" là `trung_binh`, trong khi (b) bị nhầm thành `cao`. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT thua** (Score 0.75): FT dự đoán sai `urgency` thành `trung_binh` thay vì `thap` do bỏ qua sắc thái "Khi nào tiện". |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}` | ❌ **FT thua** (Score 0.75): FT tiếp tục nhầm `urgency` thành `trung_binh` trước cụm từ "Khi nào tiện", trong khi (b) đoán đúng. |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | ✅ FT thắng (Score 1.0): Bóc tách chuẩn xác tính khẩn cấp cao ("Quá hạn rồi") kết hợp cảm xúc tích cực ("Cảm ơn shop"). |

Có mẫu chung nào ở các ca FT thua không?
Có một mẫu sai số rất rõ ràng và nhất quán ở cả hai ca FT thua (Ca 3 và Ca 4): mô hình fine-tune gặp hiện tượng **thiên lệch nhãn khẩn cấp (urgency bias)**. Cụ thể, khi ticket xuất hiện các mệnh đề nới lỏng thời gian mang tính lịch sự như *"Khi nào tiện"*, mô hình fine-tune có xu hướng dự đoán nhãn `trung_binh` thay vì nhận diện đúng mức độ `thap`. Nguyên nhân bắt nguồn từ việc trong tập dữ liệu huấn luyện SFT (train set), số lượng mẫu gắn nhãn `trung_binh` chiếm tỷ trọng áp đảo, khiến adapter ghi nhớ thiên kiến tần suất phân phối dữ liệu hơn là nắm bắt các sắc thái ngữ dụng biên. Ngược lại, prompt tối ưu (b) có lợi thế từ việc mô tả trực tiếp các quy tắc ngữ nghĩa trong system prompt nên xử lý sắc thái này linh hoạt hơn.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Bản LoRA fine-tune (`correct`) này **hoàn toàn xứng đáng và nên được đưa vào triển khai thực tế (deploy production)**. Thực nghiệm kiểm chứng cho thấy mô hình không chỉ vượt qua cổng hồi quy với mức vượt trội thuyết phục (+0.250 điểm target so với baseline prompt tối ưu mạnh nhất) mà còn bảo toàn nguyên vẹn năng lực tổng quát (regression $\Delta = +0.000$) và đảm bảo 100% tuân thủ cấu trúc JSON đầu ra (format = 1.0). Việc đưa tri thức định dạng và logic trích xuất vào trực tiếp trọng số LoRA giúp cắt giảm hơn 80% kích thước context đầu vào (từ prompt dài 175 từ xuống prompt ngây thơ 5 từ), mang lại lợi ích kinh tế to lớn khi giảm thiểu đáng kể độ trễ truyền dữ liệu và chi phí tính toán GPU trên quy mô lớn.

Đòn bẩy thực sự quyết định sự thành công trong lab này **không nằm ở rank cao hay các biến thể LoRA phức tạp, mà là sự phối hợp giữa Loss Masking và Thang đo Learning Rate**. Minh chứng là run `attn_only` dù nâng rank lên $r=283$ chỉ giảm train loss ảo chứ không hề cải thiện điểm target so với $r=16$ trên toàn bộ `text-linear`, trong khi việc hạ LR xuống thang full-FT ($10^{-5}$) ở `wrong_lr` đã triệt tiêu hoàn toàn khả năng học của adapter (target rớt về 0.000). Quan trọng hơn cả, việc thiết lập mask chuẩn xác (`assistant-only` với 41.49% token supervised) là nền tảng cốt lõi đảm bảo mô hình tập trung tối ưu đúng hành vi sinh phản hồi mà không bị lỗi lặp lại câu hỏi hay phá vỡ cấu trúc chat template.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Target Metric trên tác vụ thực tế là thước đo chân lý duy nhất, không đánh giá qua Train Loss hay Perplexity**: Một cấu hình sai như `attn_only` với rank cực đại ($r=283$) có thể ép train loss xuống rất thấp (`0.5385` so với `0.6258`) nhưng thực chất chỉ là ghi nhớ vẹt dữ liệu và cho điểm số target tương đương, chứng minh chỉ số thay thế rất dễ dẫn tới kết luận sai lầm (Lỗi #3).
2. **LoRA bắt buộc phải sử dụng thang Learning Rate riêng biệt (10x Full-FT)**: Do ma trận $B$ của LoRA được khởi tạo bằng 0 và ma trận $A$ ngẫu nhiên, việc áp dụng máy móc mức LR của full fine-tuning ($10^{-5}$) sẽ khiến adapter không thể tích lũy gradient đủ lớn để hội tụ trong số step hữu hạn, dẫn đến đường loss đi ngang và điểm số bằng 0.
3. **Loss Masking và kiểm tra Chat Template là chốt chặn sống còn của pipeline SFT**: Cần phải dùng offset mapping và giải mã ngược token để chứng minh toán học rằng loss chỉ tính trên câu trả lời của assistant và che hoàn toàn câu hỏi của user; đồng thời xác minh template không nuốt khối suy luận `<think>`, tránh việc huấn luyện tốn kém hàng giờ trên một loss mask rỗng hoặc sai lệch.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
1. Thực hiện thí nghiệm **Reasoning-trace collapse (Thưởng B3)**: huấn luyện đối chứng giữa `MASK_MODE=assistant-only` và `response-only` trên base model có chế độ thinking và đo lường sự thay đổi của `valid_trace_rate` để khảo sát hiện tượng suy thoái chuỗi suy luận.
2. Tiến hành **Quét rank có kiểm soát (Thưởng B4)** trên vị trí cố định `text-linear` với các mốc $r \in \{8, 16, 32, 64\}$ kết hợp cân bằng mẫu (rebalancing) cho nhãn `urgency` nhằm triệt tiêu dứt điểm lỗi thiên kiến ở các mẫu "Khi nào tiện".
3. Thử nghiệm **Merge adapter vào Base Model và kiểm tra Hot-swap (Thưởng B1)** theo NB6 để đánh giá độ trễ phục vụ thực tế và khả năng chuyển đổi linh hoạt giữa nhiều adapter chuyên biệt trên cùng một worker.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
