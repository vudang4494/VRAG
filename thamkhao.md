# Kế Hoạch Triển Khai Nâng Cấp & Fix Lỗi (Tham Khảo)

Tài liệu này vạch ra cấu trúc file và lộ trình cụ thể để thực hiện hai hạng mục lớn:
1. **Phase 8:** Tích hợp Semantic Domain Reward ("Màu sắc" Vector).
2. **Quick Fixes:** Khắc phục lỗi Over-refusal của ReAct Agent và Generation Prompt, cùng với kế hoạch chạy Full Benchmark.

---

## PHẦN A: TRIỂN KHAI "SEMANTIC DOMAIN REWARD" (PHASE 8)

**Mục tiêu:** Mã hóa Domain/Chủ đề thành một Vector (đại diện cho các "Màu sắc"). Sử dụng độ tương đồng giữa Vector Màu sắc của Query và Chunk làm điểm thưởng (Reward) ở bước L2R Reranking để tăng độ chính xác, tránh nhầm lẫn bối cảnh.

### 1. Cấu trúc File & Sửa đổi

*   **`src/services/domain_classifier.py`** ✨ *(Tạo mới)*
    *   **Nhiệm vụ:** Dự đoán "Màu sắc" của đoạn văn bản.
    *   **Đầu vào:** `text` (String).
    *   **Đầu ra:** `domain_vector` (Array Float, ví dụ 10 chiều tương ứng 10 chủ đề).
    *   **Phương pháp:** Có thể sử dụng Zero-shot classifier nhẹ hoặc một Prompt gọi LLM nhanh.

*   **`src/services/ingestion_v2.py`** 🛠 *(Chỉnh sửa)*
    *   **Nhiệm vụ:** Gắn "Màu sắc" vào Chunk khi lưu vào DB.
    *   **Thay đổi:** Trong quá trình xử lý Document, mỗi khi tạo ra một Chunk, gọi hàm từ `domain_classifier.py` để lấy mảng xác suất. Lưu mảng này vào trường `payload.domain_vector` khi đẩy dữ liệu lên Qdrant.

*   **`src/services/rerank_l2r.py`** 🛠 *(Chỉnh sửa)*
    *   **Nhiệm vụ:** Bổ sung Feature tính điểm thưởng.
    *   **Thay đổi:** Thêm Feature thứ 12 mang tên `domain_match_reward`.
    *   **Công thức:** `Reward = Cosine_Similarity(query_color_vector, chunk_color_vector)`. Nếu 2 vector càng giống "màu" nhau, điểm Reward càng cao và Chunk càng được đẩy lên top.

*   **`scripts/init-qdrant.sh`** 🛠 *(Chỉnh sửa)*
    *   **Nhiệm vụ:** Cập nhật Schema.
    *   **Thay đổi:** Khai báo thêm cấu trúc dữ liệu cho trường `domain_vector` trong payload của bộ sưu tập (collection).

---

## PHẦN B: KẾ HOẠCH QUICK-FIXES & FULL BENCHMARK

**Mục tiêu:** Sửa các lỗi từ chối trả lời sai (Over-refusal), tăng độ dẻo dai cho Agent và chạy đo lường 30 câu hỏi để thấy rõ hiệu năng.

### 1. Task B1: Chữa bệnh "Bỏ cuộc sớm" (Over-refusal) cho ReAct Agent
*   **File cần sửa:** `src/services/react_loop.py`
*   **Chi tiết thực hiện:** 
    *   Cập nhật biến `SYSTEM_PROMPT` của Agent.
    *   **Chỉ thị (Instruction) bắt buộc thêm vào:**
        > "CRITICAL RULE: Thao tác `FINISH` (Từ chối trả lời) là lựa chọn cuối cùng. Mặc định, KHÔNG ĐƯỢC phép gọi `FINISH` ngay sau khi tìm kiếm đồ thị thất bại. Nếu bạn gọi `expand_relation` hoặc `search_entity` mà trả về mảng kết quả rỗng (0 results), BẠN BẮT BUỘC phải gọi hành động `retrieve_chunks` (Vector Search) để rà soát toàn bộ tài liệu dự phòng trước khi được phép dùng `FINISH`."

### 2. Task B2: Nới lỏng Generation Prompt
*   **File cần sửa:** File chứa Prompt dùng để tổng hợp câu trả lời cuối cùng (thường nằm ở `src/services/validation.py`, `src/services/consistency.py` hoặc luồng xử lý chính trong file chuyên gọi LLM).
*   **Chi tiết thực hiện:**
    *   Tìm và gỡ bỏ các ràng buộc quá khắt khe (ví dụ: "chỉ trả lời nếu văn bản khớp 100% logic từng chữ").
    *   **Chỉ thị (Instruction) thay thế:**
        > "Bạn được phép tổng hợp, so sánh và diễn giải dựa trên ý nghĩa ngữ cảnh (semantic meaning) của các nguồn tài liệu cung cấp. Miễn là thông tin đưa ra không mâu thuẫn với sự thật trong nguồn và không bịa đặt sự thật mới (hallucination), hãy đưa ra câu trả lời chi tiết và đầy đủ nhất thay vì từ chối."

### 3. Task B3: Chạy & Cập nhật Full Benchmark
*   **Chi tiết thực hiện:**
    *   Sử dụng Terminal chạy lại script đánh giá với toàn bộ 30 câu hỏi để thấy sức mạnh của GAEA và ReAct trên các câu `multi_hop` và `summarization`.
    *   **Lệnh thực thi:**
        ```bash
        python3 scripts/ablation_eval.py \
          --bench eval/datasets/vi_benchmark_v1.json \
          --tenant eval \
          --output eval/results/ablation_full_run.json
        ```
    *   **Cập nhật Báo Cáo:** Lấy số liệu tổng hợp (Aggregate Results) in ra từ Terminal và cập nhật/ghi đè vào file `.claude/internal-docs/reports/PHASE_7_BENCHMARK_RESULTS.md` để lưu lại minh chứng hiệu năng sau khi đã áp dụng các Quick-Fixes.
