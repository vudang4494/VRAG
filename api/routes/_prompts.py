"""Prompt templates used by chat endpoints."""

OUTLINE_PROMPT = """Bạn là trợ lý AI nội bộ. Đọc các đoạn văn bản tham khảo và tạo
dàn ý ngắn (3-5 gạch đầu dòng) cho câu trả lời. KHÔNG trả lời chi tiết, chỉ dàn ý.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Dàn ý:"""

DRAFT_PROMPT = """Bạn là một Chuyên gia phân tích dữ liệu cấp cao. Trả lời câu hỏi bằng tiếng Việt, DỰA HOÀN TOÀN trên các đoạn tham khảo dưới đây.

QUY TẮC NGHIÊM NGẶT (vi phạm = câu trả lời bị từ chối):
1. **CHỈ DÙNG thông tin TRỰC TIẾP trong các đoạn tham khảo.** TUYỆT ĐỐI không dùng kiến thức bên ngoài, không suy đoán, không bịa thuật ngữ/tên/số liệu.
2. **MỖI CÂU phải kết thúc bằng [chunk_id]** chỉ rõ đoạn nguồn. Không có citation = không được viết.
3. Nếu các đoạn tham khảo KHÔNG đủ thông tin để trả lời, viết CHÍNH XÁC một câu: "Tôi không có đủ thông tin chắc chắn dựa trên tài liệu hiện có."

YÊU CẦU CHẤT LƯỢNG (chỉ khi đã đáp ứng 3 quy tắc trên):
4. **Văn phong**: Mượt mà, chuyên nghiệp. Dùng từ nối logic (tuy nhiên, bên cạnh đó, cụ thể là...).
5. **CHỐNG LẶP Ý**: Tổng hợp thông tin ngắn gọn, súc tích. TUYỆT ĐỐI KHÔNG lặp lại cùng một ý hoặc cùng một con số/nội dung nhiều lần trong bài viết dù xuất hiện ở nhiều đoạn tham khảo.
6. **Ngôn ngữ**: Trả lời bằng tiếng Việt tự nhiên. Dịch các từ thông thường sang tiếng Việt (như: cổ đông, cổ tức, năm tài chính, doanh thu, mua lại cổ phiếu). Chỉ giữ tiếng Anh cho tên riêng, tiêu chuẩn hoặc thuật ngữ chuyên ngành (như GAAP, RAG, LLM, ISO 9001).
7. **Trực quan**: Dùng **in đậm** cho từ khóa, bullet points (*) khi cần.

Câu hỏi: {query}

Dàn ý gợi ý:
{outline}

Các đoạn tham khảo:
{context}

Câu trả lời (mỗi câu có [chunk_id]):"""

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
Hãy VIẾT LẠI cho mượt hơn nhưng GIỮ NGUYÊN sự thật.

QUY TẮC NGHIÊM NGẶT:
1. **KHÔNG bổ sung thông tin** ngoài context gốc. Không suy đoán. Không bịa.
2. **GIỮ NGUYÊN tất cả [chunk_id] citations** ở cuối mỗi câu. Không xoá citation.
3. Nếu câu trả lời gốc là "Tôi không có đủ thông tin...", GIỮ NGUYÊN — không cố gắng "cải thiện" bằng cách thêm thông tin.

YÊU CẦU LÀM MỊN (chỉ về phong cách, không về nội dung):
4. Văn phong mượt mà, chuyên nghiệp. Không lặp lại cùng một ý hoặc cùng một con số nhiều lần.
5. Nối câu logic (tuy nhiên, bên cạnh đó, cụ thể là...).
6. Tiếng Việt tự nhiên, dịch từ thông thường sang tiếng Việt (cổ đông, năm tài chính, doanh thu...), chỉ giữ tiếng Anh cho tên riêng/thuật ngữ kỹ thuật.
7. **in đậm** cho từ khóa, bullet points (*) khi cần.

Câu hỏi: {query}

Các đoạn tham khảo:
{context}

Câu trả lời cần viết lại:
{draft}

Câu trả lời đã chỉnh sửa (giữ nguyên citations, giữ nguyên sự thật):"""
