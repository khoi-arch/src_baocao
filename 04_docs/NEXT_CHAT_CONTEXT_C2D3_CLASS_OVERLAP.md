# Context chuyển sang chat mới — Official C2+D3 và vấn đề class overlap

## 0. Mục tiêu của repo mới

Repo mới `src_baocao` là repo chính thức, không còn là repo thí nghiệm lộn xộn. Từ bây giờ:

- `C2` tokenization + `D3` model được xem là **official baseline / pipeline mặc định**.
- Các tên quá dài kiểu `K512_B512_C2_selective_rank_discrete_compact` hoặc `D3_P1_K512...` chỉ nên giữ trong lịch sử hoặc metadata, không nên để làm cấu trúc chính của repo.
- Mục tiêu tiếp theo không phải tiếp tục vá triệu chứng nhỏ, mà là xử lý vấn đề cốt lõi: **các malware class overlap và model hiện tại chưa phân biệt đủ tốt giữa Ransomware / Spyware / Trojan**.

Repo mới hiện cần có cấu trúc gọn kiểu:

```text
00_raw_dataset/
01_split/
02_src/
03_outputs/
04_docs/
```

Trong đó `02_src` chỉ giữ code chính thức, còn `03_outputs` chỉ giữ artifact cần thiết để chứng minh official baseline và audit class-overlap.

---

## 1. Official baseline hiện tại

### 1.1. Tokenization chính thức

Official tokenization là C2:

```text
K512 + B512 + selective rank + discrete compact
```

Ý nghĩa chính:

- Một nhóm feature high-compression/high-unique dùng `rank_uniform_offset`.
- Một nhóm low-unique/discrete/binary dùng `discrete_compact_offset0` để tránh offset noise.
- Một nhóm `keep_current` giữ cách encode hiện tại vì audit cho thấy nó chứa signal quan trọng.
- Một số feature constant được xử lý riêng.

Kết luận từ các thử nghiệm cũ: C2 là tokenization ổn nhất hiện tại. Các thay đổi thô như tăng K global, compact toàn bộ `keep_current`, hoặc merge rare token đều không giải quyết được vấn đề.

### 1.2. Model chính thức

Official model là D3:

```text
shared bin embedding
+ offset interpolation
+ raw continuous FiLM/gating
+ feature embedding
+ Transformer over [CLS] + feature tokens
+ classifier từ CLS
```

Cơ chế chính:

```python
local = (1 - offset) * Emb(bin) + offset * Emb(bin + 1)
value_emb = local * (1 + gate * gamma(raw_scaled)) + gate * beta(raw_scaled)
cell_emb = concat([value_emb, feature_embedding[f]])
CLS = Transformer([CLS] + feature_tokens)
logits = classifier(CLS)
```

Shared bin embedding không phải bug. Test full per-feature embedding đã fail, cho thấy shared embedding đang đóng vai trò regularizer.

### 1.3. Official metric cần nhớ

Historical official C2+D3:

```text
val macro-F1 ≈ 0.817147
```

Trong lần recompute official sau khi sửa version:

```text
accuracy   ≈ 0.878584
macro-F1   ≈ 0.817281
top2 acc   ≈ 0.968857
```

Hai số macro-F1 này đủ sát nhau, có thể xem là cùng official baseline.

---

## 2. Vấn đề cốt lõi đã rút ra

### 2.1. Không phải chủ yếu Benign-vs-Malware

Benign gần như đã ổn. Lỗi chính nằm ở nội bộ malware subtype:

```text
Ransomware ↔ Spyware
Ransomware ↔ Trojan
Spyware ↔ Trojan
```

Do đó các hướng kiểu chỉ tăng regularization, giảm confidence, hoặc chỉnh calibration không giải quyết được gốc rễ.

### 2.2. Class overlap là bottleneck chính

Audit raw/token/CLS cho thấy:

- Trong raw/token space, các malware subtype overlap mạnh nhưng không hoàn toàn trùng nhau.
- Top-2 accuracy rất cao, nghĩa là true class thường vẫn nằm trong hai ứng viên đầu.
- Trong CLS/logit decision, ambiguous samples dễ bị kéo mạnh về predicted class centroid.
- Sai lầm thường xảy ra khi model phải chọn giữa hai malware subtype gần nhau.

Official A0 sau khi sửa version cho thấy:

```text
top1 accuracy = 0.878584
top2 accuracy = 0.968857
wrong_total = 1423
wrong_true_in_top2 = 1058
wrong_true_in_top2_rate ≈ 74.35%
```

Điều này rất quan trọng:

```text
Model thường đã đưa true class vào top-2.
Vấn đề không phải thiếu hoàn toàn tín hiệu.
Vấn đề là model chưa phân xử đủ tốt giữa các malware class overlap.
```

### 2.3. Diễn giải root cause

Root cause hiện tại nên viết ngắn gọn như sau:

```text
C2+D3 đã học được phân biệt malware/benign và có top-2 coverage rất cao,
nhưng chưa học được boundary đủ sắc giữa các malware subtype overlap.
Các class Ransomware, Spyware, Trojan có vùng giao nhau trong raw/token space;
encoder/CLS giữ một phần tín hiệu nhưng decision top-1 vẫn thường collapse về class sai.
Vì vậy các biện pháp chữa triệu chứng như confidence smoothing, center loss, regularization,
merge rare token, hoặc rerank tuyến tính đơn giản đều không giải quyết tận gốc.
```

---

## 3. Các test cũ đã thử nhưng không hiệu quả

### 3.1. Tokenization / preprocessing ablation

Các hướng đã thử:

- Tăng K global lên K1024.
- Rank-safe K1024.
- Absolute-control K1024.
- Compact hoặc offset-off nhóm `keep_current`.
- Rare physical merge cho nhóm `keep_current`.

Kết luận:

```text
Không giải quyết class overlap.
Một số metric token đẹp hơn nhưng model F1 giảm.
Tăng K hoặc merge rare token thô có thể làm mất signal cục bộ.
C2 vẫn là tokenization tốt nhất hiện tại.
```

### 3.2. Embedding / architecture ablation

Các hướng đã thử:

- Full per-feature embedding.
- Feature adapter / interaction mixer nhẹ.
- D8/D8F/PFE/LSTM/MHA backbone variants.
- Residual/smoothness D3S2R.
- Raw branch variants: raw minmax, log/rank/clip/blend.

Kết luận:

```text
Không có hướng nào vượt ổn định C2+D3.
Full per-feature embedding còn tệ hơn, chứng tỏ shared bin embedding không phải lỗi.
Backbone phức tạp hơn không tự giải quyết malware subtype overlap.
```

### 3.3. Regularization / confidence-only

Các hướng đã thử:

- Label smoothing.
- Weight decay variants.
- Dropout/no class weights.
- Calibrated/lower loss variants.

Kết luận:

```text
Có thể làm confidence/loss đẹp hơn một chút nhưng không sửa boundary overlap.
Một số run tăng rất nhỏ hoặc không ổn định, không giải quyết root cause.
```

### 3.4. Boundary-only center loss

Đã thử center loss để kéo representation cùng class lại gần hơn.

Kết luận:

```text
Center loss giảm shift/confidence nhưng macro-F1 giảm.
Lý do: malware class có thể multi-mode; ép về một center/class làm mất cấu trúc subtype nhỏ.
```

### 3.5. Pairwise reranker A

Mục tiêu ban đầu: kiểm tra liệu top-2 có đủ tín hiệu để rerank không.

A0 official cho thấy top-2 oracle rất cao, nên hướng dispute là có cơ sở.

Nhưng cần lưu ý:

- Một số kết quả A/B ban đầu bị mixed-version: CLS lấy từ official C2+D3 nhưng A0 routing lấy từ Test1 control. Các kết quả đó không được dùng làm kết luận cuối.
- Sau khi sửa version, B2 official cho thấy frozen CLS prototype rerank không thắng ổn định.

Kết luận sạch:

```text
Top-2 oracle rất cao → còn dư địa phân xử.
Nhưng reranker đơn giản trên CLS/prototype không khai thác được oracle gap.
```

### 3.6. Multi-prototype / family-mixture B

Mục tiêu: xem mỗi malware class có nhiều mode trong CLS không, và dùng nhiều prototype/class có phân xử top-2 tốt hơn không.

Kết quả official B2:

```text
Global K hoặc pair-specific K đều không thắng base ổn định.
Best pair-specific vẫn âm hoặc chỉ tăng rất nhỏ nếu post-hoc.
Không đủ để claim hướng B là lời giải.
```

Kết luận:

```text
Frozen CLS có thể có cấu trúc cluster hữu ích,
nhưng prototype-distance reranking không đủ để giải quyết boundary overlap thực tế.
Không nên tiếp tục refine B quá lâu.
```

---

## 4. Bài học chính sau toàn bộ audit/test

### 4.1. Không nên tiếp tục chữa triệu chứng

Các hướng sau không nên ưu tiên tiếp:

```text
- Chỉ giảm confidence.
- Chỉ thêm regularization.
- Chỉ center loss một centroid/class.
- Chỉ tăng K tokenization global.
- Chỉ merge rare token.
- Chỉ rerank tuyến tính hoặc prototype frozen CLS.
```

Vì chúng không tackle đúng vấn đề:

```text
malware subtype overlap + decision boundary chưa đủ tốt giữa các class malware gần nhau.
```

### 4.2. Tín hiệu vẫn tồn tại nhưng khó khai thác

Top-2 oracle cao chứng minh:

```text
true class thường vẫn còn trong candidate set.
```

Nhưng các test đơn giản fail chứng minh:

```text
micro-signal không nằm ở dạng tuyến tính đơn giản,
cũng không được khai thác đủ bằng frozen CLS prototype.
```

Do đó hướng tiếp theo phải học trực tiếp hơn các pattern phân biệt overlap malware.

---

## 5. Plan test mới để tackle class overlap

### Nguyên tắc làm việc

Mỗi test phải:

1. Có mô tả trước khi code: test kiểm tra giả thuyết gì, input gì, expected outcome gì.
2. Chỉ thay đổi một nhóm cơ chế mỗi lần.
3. Không trộn nhiều ý trong một run.
4. Xuất đầy đủ output:
   - metrics tổng thể,
   - per-class F1,
   - malware-only F1,
   - confusion matrix,
   - pairwise confusion,
   - repair/damage nếu có rerank,
   - top-2/oracle audit,
   - corrected vs broken sample audit.
5. Không dùng val để chọn rule rồi claim chính thức. Nếu chọn trên val thì ghi rõ diagnostic.

### Test C — Micro-interaction / nonlinear pairwise branch

Giả thuyết:

```text
Raw/token linear fail vì micro-signal nằm ở interaction phi tuyến giữa feature hành vi,
ví dụ quan hệ giữa handles, malfind, ldrmodules, dlllist, psxview.
```

Không nên chỉ logistic trên raw/token thô. Cần test interaction có kiểm soát:

- diff / absdiff / ratio / product giữa feature cùng group.
- aggregate relation giữa nhóm feature.
- domain interaction cố định, không chọn theo val.
- pairwise malware heads cho RS/RT/ST.

Mục tiêu:

```text
Xem interaction raw/token có cứu được các top-2 disputed samples không,
đặc biệt các cặp mà CLS prototype fail.
```

### Test D — Hard-pair anti-amplification training

Giả thuyết:

```text
Trong training, encoder/classifier học decision quá tự tin ở các vùng overlap.
Cần loss trực tiếp cho hard malware pairs thay vì confidence smoothing chung.
```

Ý tưởng:

- Không dùng center loss một centroid/class.
- Tạo hard-pair objective trên các cặp Ransomware/Spyware/Trojan.
- Tăng margin đúng giữa true class và confusing malware class.
- Có thể dùng pairwise auxiliary head hoặc hard-negative margin loss.

Mục tiêu:

```text
Giảm collapse/amplification ở vùng overlap mà không ép mỗi class về một center.
```

### Test E — Multi-task L2/L3 hoặc subtype-aware training nếu có label L3

Giả thuyết:

```text
L2 malware class quá thô; mỗi L2 gồm nhiều subtype/mode.
Dùng L3/subfamily supervision có thể giúp representation phân biệt tốt hơn.
```

Ý tưởng:

- Train auxiliary head dự đoán L3/subfamily nếu label còn trong raw dataset.
- Main head vẫn là L2.
- Không dùng L3 ở inference nếu mục tiêu cuối là L2, nhưng dùng để shape representation.

Mục tiêu:

```text
Giúp model học cấu trúc nội bộ của từng malware family thay vì chỉ một nhãn L2 thô.
```

### Test F — Pair-specific decision head integrated vào model

Giả thuyết:

```text
Global 4-class head không đủ tốt cho các boundary pairwise.
Nhưng frozen reranker fail vì không được train end-to-end với representation.
```

Ý tưởng:

- Giữ global 4-class head.
- Thêm 3 pairwise heads trong training.
- Loss phụ chỉ áp dụng cho malware pair samples.
- Inference có thể dùng global top-2 để gọi pair head, hoặc dùng auxiliary consistency loss trước.

Mục tiêu:

```text
Ép CLS giữ tín hiệu phân biệt từng cặp malware ngay trong quá trình training,
thay vì rerank frozen representation sau đó.
```

---

## 6. Repomix command để tạo XML cho chat mới

Chạy trong repo mới `src_baocao` sau khi đã dọn layout:

```bash
cd ~/Documents/src_baocao

npx repomix@latest . \
  --style xml \
  --output official_c2d3_class_overlap_context.xml \
  --include "README.md,README_OFFICIAL_C2D3.md,requirements.txt,02_src/**/*.py,03_outputs/00_dataset/metadata.json,03_outputs/01_model/config.json,03_outputs/01_model/diagnosis_summary.json,03_outputs/01_model/history.csv,03_outputs/01_model/reports/**/*.json,03_outputs/01_model/reports/**/*.csv,03_outputs/01_model/predictions/val_predictions*.csv,03_outputs/02_audit_best/**/*.json,03_outputs/02_audit_best/**/*.csv,03_outputs/02_audit_best/**/*.md,03_outputs/03_audit_rootcause/**/*.json,03_outputs/03_audit_rootcause/**/*.csv,03_outputs/03_audit_rootcause/**/*.md,04_docs/**/*.md" \
  --ignore "**/*.pt,**/*.npz,**/*.zip,00_raw_dataset/**,01_split/*.csv,_backup*/**,.archive/**,__pycache__/**,.git/**"
```

Nếu repo vẫn còn layout cũ chưa đổi tên, dùng command fallback này:

```bash
cd ~/Documents/src_baocao

npx repomix@latest . \
  --style xml \
  --output official_c2d3_class_overlap_context.xml \
  --include "README.md,README_OFFICIAL_C2D3.md,requirements.txt,02_src/**/*.py,03_outputs/build_mixed_quantile_offset/K512_B512_C2_selective_rank_discrete_compact/mixed_quantile_offset_metadata.json,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/config.json,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/diagnosis_summary.json,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/history.csv,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/*classification_report*.json,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/*confusion_matrix*.csv,03_outputs/train_runs_fusion_ablation_D0_D7/Keff512/D3_P1_K512_B512_C2_selective_rank_discrete_compact/val_predictions_best.csv,03_outputs/audit_c2_best/**/*.json,03_outputs/audit_c2_best/**/*.csv,03_outputs/audit_c2_best/**/*.md,03_outputs/audit_overfit_rootcause/**/*.json,03_outputs/audit_overfit_rootcause/**/*.csv,03_outputs/audit_overfit_rootcause/**/*.md,04_docs/**/*.md" \
  --ignore "**/*.pt,**/*.npz,**/*.zip,00_raw_dataset/**,01_split/*.csv,_backup*/**,.archive/**,__pycache__/**,.git/**"
```

Sau đó sang chat mới gửi:

```text
1. File official_c2d3_class_overlap_context.xml
2. File markdown này
3. Nói rõ: bắt đầu từ repo mới official C2+D3, cần tackle class overlap giữa malware classes.
```

---

## 7. Câu mở đầu gợi ý cho chat mới

```text
Tôi đang làm lại repo chính thức cho bài toán CIC-MalMem L2 4-class.
Official baseline là C2 tokenization + D3 model, macro-F1 khoảng 0.817.
Các test cũ cho thấy vấn đề cốt lõi không phải Benign-vs-Malware mà là class overlap giữa Ransomware/Spyware/Trojan.
Top-2 accuracy khoảng 0.969, tức true class thường nằm trong top-2 nhưng top-1 phân xử sai.
Tôi muốn tiếp tục từ context XML và markdown này, trước hết hãy đọc repo/context rồi giúp tôi thiết kế Test C để xử lý overlap bằng interaction/nonlinear pairwise mechanism.
Nguyên tắc: trước khi code phải mô tả test, mỗi lần chỉ test một ý, output phải có đầy đủ metrics và audit.
```
