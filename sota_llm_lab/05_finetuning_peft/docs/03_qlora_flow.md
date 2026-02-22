# 05.3 – Luồng QLoRA (Quantized LoRA)

- **Mục đích**: Train LoRA trên base model đã **quantize 4-bit** (QLoRA) để giảm VRAM/RAM so với 05.2.
- **Script**: `src/train_qlora.py`.

## Cách chạy

```bash
cd src
python train_qlora.py
```

- **Có GPU**: Cần `pip install bitsandbytes`. Model load 4-bit (NF4), train LoRA, lưu adapter tại `experiments/qlora_run/final_adapter`.
- **Chỉ CPU / Windows**: Tự fallback LoRA full precision (không quantize), vẫn lưu vào `experiments/qlora_run` để bước 05.5 so sánh với 05.2.

## So với 05.2 (LoRA)

|        | 05.2 LoRA   | 05.3 QLoRA   |
|--------|-------------|--------------|
| Base   | FP16/FP32   | 4-bit (khi có GPU) |
| VRAM   | Cao hơn     | Thấp hơn     |
| Chất lượng | Tham khảo | So sánh ở 05.5 |

## Cấu hình trong script

- `USE_4BIT = True`: dùng 4-bit (QLoRA); `False`: 8-bit.
- Các hằng LoRA (r, alpha, target_modules) giống 05.2; có thể chỉnh trong file.
