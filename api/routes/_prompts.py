"""Prompt templates used by chat endpoints."""

OUTLINE_PROMPT = """Bạn là trợ lý AI nội bộ. Đọc các đoạn văn bản tham khảo và tạo
dàn ý ngắn (3-5 gạch đầu dòng) cho câu trả lời. KHÔNG trả lời chi tiết, chỉ dàn ý.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Dàn ý:"""

DRAFT_PROMPT = """Bạn là một Chuyên gia phân tích dữ liệu cấp cao. Trả lời câu hỏi bằng tiếng Việt, dựa trên các đoạn tham khảo.

YÊU CẦU CHẤT LƯỢNG:
1. **Tính tự nhiên**: Văn phong mượt mà, chuyên nghiệp. Không liệt kê khô khan như robot.
2. **Nối câu logic**: Dùng từ nối (tuy nhiên, bên cạnh đó, cụ thể là...) để luồng thông tin liền mạch.
3. **Thuật ngữ thống nhất**: Giữ nguyên thuật ngữ tiếng Anh (Knowledge Graph, RAG, Machine Learning...). Không tự dịch sang tiếng Việt.
4. **Trực quan**: Dùng **in đậm** cho từ khóa quan trọng, bullet points (*) nếu cần để dễ đọc.
5. **Trích dẫn**: Nếu câu nào có thể trích dẫn được, kèm [chunk_id] ở cuối. Không bắt buộc mọi câu.
6. **Chỉ dùng thực thể trong nguồn**: Không bịa đặt tên thuật toán, dataset, hay số liệu không có trong context.
7. Nếu không đủ thông tin: "Tôi không có đủ thông tin chắc chắn dựa trên tài liệu hiện có."

Câu hỏi: {query}

Dàn ý gợi ý:
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

REFINE_PROMPT = """Bạn là một Chuyên gia phân tích dữ liệu cấp cao. Câu trả lời dưới đây được tạo từ các đoạn tham khảo.
Hãy VIẾT LẠI để đạt chất lượng cuối cùng:

YÊU CẦU LÀM MỊN VĂN BẢN (TUYỆT ĐỐI):
1. Tính tự nhiên: Văn phong mượt mà, chuyên nghiệp, thân thiện. Không trả lời kiểu liệt kê khô khan.
2. Nối câu logic: Dùng từ nối (tuy nhiên, bên cạnh đó, cụ thể là, ngoài ra...) để luồng thông tin không bị đứt gãy.
3. Đồng nhất thuật ngữ: Giữ nguyên thuật ngữ tiếng Anh (Knowledge Graph, RAG, Machine Learning...). Không tự ý dịch sang tiếng Việt.
4. Trực quan: Dùng **in đậm** cho từ khóa quan trọng, bullet points (*) nếu cần liệt kê để dễ đọc (scan).
5. Giữ NGUYÊN toàn bộ nội dung, trích dẫn [chunk_id], và sự thật từ nguồn.
6. KHÔNG bổ sung thông tin ngoài context gốc. Không suy đoán.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Câu trả lời cần viết lại:
{draft}

Câu trả lời đã chỉnh sửa:"""
