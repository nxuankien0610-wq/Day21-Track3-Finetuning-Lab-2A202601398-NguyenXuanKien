# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Sự "đánh lừa" của Train Loss đối với năng lực thực tế của mô hình: Run `attn_only` khi được đẩy rank lên $r=283$ (để khớp ngân sách 32.4M tham số với `correct`) đã ép train loss xuống rất thấp (`0.5385` so với `0.6258` của `correct`), nhưng khi đo trên tác vụ thực tế (target) thì chỉ đạt kết quả hòa (`0.9375`). Điều này chứng minh rằng rank cực lớn ở vị trí hẹp chỉ giúp adapter ghi nhớ dữ liệu huấn luyện (memorization) chứ không hề gia tăng khả năng tổng quát hóa so với việc phân bổ rank $r=16$ trên toàn bộ các lớp `text-linear`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Mất nhiều thời gian nhất ở bước **NB1 (kiểm chứng loss mask và chat template)** và thời gian chờ huấn luyện 3 cấu hình đối chứng ở NB4 (~45–60 phút trên GPU). Ban đầu tôi dự đoán thời gian sẽ tiêu tốn nhiều nhất cho việc viết vòng lặp huấn luyện hoặc tinh chỉnh siêu tham số phức tạp, nhưng thực tế việc giải mã ngược token để chứng minh toán học rằng loss chỉ phủ lên phần trả lời của assistant và đảm bảo template không nuốt khối `<think>` mới là khâu đòi hỏi sự tỉ mỉ nhất để tránh huấn luyện sai từ gốc.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin vào 3 điều phổ biến: (1) Cứ tăng rank $r$ lên thật cao thì LoRA sẽ càng thông minh; (2) Chỉ cần gắn LoRA vào $q\_proj, v\_proj$ là đủ; và (3) Đánh giá fine-tune thành công dựa vào Perplexity hoặc Loss giảm. Sau lab này, tôi hiểu rằng: gắn adapter toàn diện trên `text-linear` với rank vừa phải ($r=16$) vượt trội hơn nhiều; đòn bẩy thực sự nằm ở Learning Rate ($10\times$ Full-FT) và Loss Masking; và nguồn chân lý duy nhất là Target Metric được so sánh công bằng với một Baseline Prompt tối ưu.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để hỗ trợ phân tích kiến trúc model Qwen3.5, kiểm tra sự tương thích tham số cấu hình TRL/PEFT, và trích xuất phân tích mẫu lỗi định tính. Chỗ AI thường mắc sai sót là: (1) Hay gợi ý cú pháp lỗi thời như `tokenizer=` thay vì `processing_class=` trong TRL mới; (2) Có xu hướng tự động đề xuất dùng QLoRA (4-bit) để tiết kiệm VRAM mà không cảnh báo việc QLoRA gây sụt giảm chất lượng trên dòng Qwen3.5; và (3) Gợi ý dùng learning rate quá nhỏ ($10^{-5}$ của full-FT) cho LoRA.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm không phải là mở GPU để train ngay, mà là **xây dựng bộ dữ liệu đánh giá chuẩn (gồm Target task và Regression set) và đo đạc Baseline (b) với Prompt tối ưu hóa**. Tôi sẽ kiểm tra xem liệu bài toán của khách hàng có thể giải quyết triệt để bằng Prompt Engineering hay không. Chỉ khi prompt tối ưu bộc lộ giới hạn về độ chính xác, format, hoặc chi phí context token/latency quá cao, tôi mới tiến hành fine-tuning và dùng chính baseline (b) này làm thước đo bắt buộc phải vượt qua.
