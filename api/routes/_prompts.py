"""Prompt templates used by chat endpoints."""

OUTLINE_PROMPT = """Bạn là trợ lý AI nội bộ. Đọc các đoạn văn bản tham khảo và tạo
dàn ý ngắn (3-5 gạch đầu dòng) cho câu trả lời. KHÔNG trả lời chi tiết, chỉ dàn ý.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Dàn ý:"""

DRAFT_PROMPT = """Bạn là trợ lý AI nội bộ. Trả lời câu hỏi sau bằng tiếng Việt,
NGẮN GỌN (3-7 câu), dựa trên các đoạn văn bản tham khảo bên dưới.

QUY TẮC (TUÂN THỦ TUYỆT ĐỐI):
1. Nếu câu nào có thể trích dẫn được, thì kèm [chunk_id] ở cuối câu đó.
   Không bắt buộc mọi câu đều phải có trích dẫn.
2. CHỈ dùng tên thực thể (entity) XUẤT HIỆN trong các đoạn tham khảo.
3. Bạn được phép tổng hợp, so sánh và diễn giải dựa trên ý nghĩa ngữ cảnh (semantic meaning).
   Miễn là thông tin đưa ra không mâu thuẫn với sự thật trong nguồn và không bịa đặt sự thật mới (hallucination),
   hãy đưa ra câu trả lời chi tiết và đầy đủ nhất thay vì từ chối.
4. Nếu thật sự không có thông tin trong đoạn tham khảo, chỉ trả lời:
   "Tôi không có đủ thông tin chắc chắn dựa trên tài liệu hiện có."

Câu hỏi: {query}

Dàn ý gợi ý (tham khảo):
{outline}

Các đoạn tham khảo:
{context}

Câu trả lời:"""

JUDGE_PROMPT = """Trong 3 bản trả lời sau, bản nào CHÍNH XÁC NHẤT (dựa trên context),
RÕ RÀNG NHẤT, và có TRÍCH DẪN ĐẦY ĐỦ nhất? Trả lời CHỈ số 1, 2, hoặc 3.

Câu hỏi: {query}

Bản 1:
{d1}

Bản 2:
{d2}

Bản 3:
{d3}

Số bản tốt nhất:"""

REFINE_PROMPT = """Bạn là biên tập viên chuyên nghiệp. Câu trả lời dưới đây được tạo từ các đoạn tham khảo.
Hãy VIẾT LẠI câu trả lời để:
1. Sửa lỗi ngữ pháp, chính tả (nếu có)
2. Câu văn mượt mà, tự nhiên hơn, KHÔNG rời rạc
3. Giữ NGUYÊN toàn bộ nội dung và trích dẫn [chunk_id] ở cuối mỗi câu
4. KHÔNG bổ sung thông tin ngoài context gốc
5. Giữ giọng văn chuyên nghiệp, thân thiện

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Câu trả lời cần viết lại:
{draft}

Câu trả lời đã chỉnh sửa:"""
