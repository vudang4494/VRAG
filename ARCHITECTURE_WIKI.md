# Enterprise Local RAG Stack v3.0 - Architecture & Algorithms

Bản thiết kế kỹ thuật (Technical Wiki) cho hệ thống **Enterprise Local RAG Stack v3.0**. Tài liệu này đi sâu vào phân tích các cấu trúc dữ liệu, luồng hệ thống và các thuật toán cốt lõi được áp dụng để đạt được hiệu suất RAG cấp độ doanh nghiệp.

## 1. Tổng quan Kiến trúc (Architecture Overview)

Hệ thống được thiết kế theo mô hình **Hybrid GraphRAG**, tận dụng tối đa sức mạnh của cả cơ sở dữ liệu Vector (Qdrant) để tìm kiếm ngữ nghĩa bề mặt và cơ sở dữ liệu Đồ thị (Neo4j) để tìm kiếm cấu trúc và quan hệ phức tạp.

**Các module cốt lõi:**
- **Vector Store**: Qdrant (Lưu trữ multi-view embeddings & BM25)
- **Knowledge Graph**: Neo4j (Lưu trữ Entity, Relationship & Community)
- **Cơ chế suy luận**: ReAct Agent cho các truy vấn Multi-hop.
- **Isolate & Cache**: Redis (Semantic cache) & Tenant Isolation toàn diện.

---

## 2. Thuật toán Ingestion & Knowledge Graph

Quy trình chuẩn bị dữ liệu (Ingestion) không chỉ dừng ở việc cắt văn bản (chunking) mà áp dụng các thuật toán phân tích đa chiều:

- **Hierarchical Chunking & 5-View Consistency:** Tài liệu được phân rã theo cấp bậc (Section -> Paragraph -> Sentence). Sau đó, một thuật toán mô phỏng (Consistency Simulation) sẽ đánh giá dữ liệu qua 5 góc nhìn: `dense`, `paraphrase`, `question`, `summary`, và `keywords`.
- **Zero-shot NER với GLiNER:** Tối ưu hóa tài nguyên bằng cách sử dụng mô hình chuyên dụng GLiNER để trích xuất thực thể (Entities) ban đầu thay vì dùng LLM trực tiếp.
- **3-Pass LLM Voting:** Thuật toán đồng thuận 3 vòng được sử dụng để xác minh và thiết lập Mối quan hệ (Relationships) giữa các Entities, đảm bảo Precision của Knowledge Graph ở mức cao nhất.
- **Leiden/Louvain Community Detection:** Áp dụng thuật toán phân cụm đồ thị phân cấp (Leiden) tương tự kiến trúc **Microsoft GraphRAG**. Thuật toán gom nhóm các Entities thành các *Community*, sau đó LLM tóm tắt từng cụm. Điều này giải quyết bài toán "Global Context" (Tóm tắt tổng thể) xuất sắc.

---

## 3. Kiến trúc Truy vấn & Fusion (9-Path Retrieval)

Đây là "trái tim" của hệ thống, giúp đối phó với mọi loại truy vấn phức tạp:

- **Multi-view Semantic Search:** Giải quyết độ lệch từ vựng (Vocabulary Mismatch) bằng cách query song song trên 5 vector ngữ nghĩa khác nhau + 1 sparse vector (BM25) từ Qdrant.
- **Graph-based Search:** Bổ sung Entity Pivot (tìm qua thực thể mỏ neo) và Community Summaries (lấy tóm tắt cụm đồ thị) từ Neo4j.
- **Weighted Reciprocal Rank Fusion (RRF):** Công thức tùy chỉnh thay vì RRF thuần túy:
  ```text
  RRF_score = path_weight × consistency_factor × level_factor × domain_reward / (k + rank)
  ```
  Hệ thống chủ động gán trọng số cao cho các path có độ tin cậy tuyệt đối (ví dụ: `Hyde=1.3`, `Entity Pivot=1.5`) và phạt điểm đối với các chunk ngữ nghĩa kém.

---

## 4. Thuật toán Suy luận (Reasoning via ReAct Loop)

Để vượt qua giới hạn tìm kiếm tĩnh (Static Retrieval), hệ thống trang bị thuật toán **ReAct Agent** (Reasoning and Acting) kết hợp Graph Traversal.

- **Dynamic Traversal:** Agent có quyền sử dụng các Tools (`search_entity`, `expand_relation`) để "đi dạo" trên đồ thị Neo4j.
- **Multi-hop Resolution:** Giúp giải quyết các truy vấn vắt chéo nhiều tài liệu. Ví dụ: *"Công ty mà ông A đang làm việc có sản phẩm gì?"*. Agent sẽ đi từ Entity `Ông A` -> `Công ty` -> `Sản phẩm`.
- *Khả năng mở rộng:* Thuật toán được thiết kế để scale tốt trên môi trường GPU Server bằng cách tăng `sample size` (số chunk fetch mỗi vòng lặp) nhằm tối đa hóa tỷ lệ Recall.

---

## 5. Thuật toán Bảo vệ & Validation (Safety Gates)

Ngăn chặn ảo giác (Hallucination) là ưu tiên số một của hệ thống Enterprise:

- **Mixed-Signal Out-of-Domain (OOD) Detection:** 
  Kết hợp **Dense Distance** (Ngưỡng Cosine < 0.5) và **Lexical Overlap** (Tỷ lệ khớp từ khóa < 30%). Thiết kế này chặn đứng "ảo giác mờ", từ chối trả lời nếu câu hỏi hoàn toàn không liên quan đến cơ sở dữ liệu.
- **Parallel Validation Gates:** Trải qua 3 cổng gác song song:
  1. **Hallucination Gate:** Bóc tách thành các claim nhỏ (Atomic Claims) và check tỉ lệ grounded ratio (phải >= 0.70).
  2. **Entity Gate:** Đối chiếu các thực thể có trong câu trả lời với Knowledge Graph.
  3. **Citation Gate:** Ép buộc LLM phải trích dẫn (Citation Ratio >= 0.40). Trả lời không có nguồn sẽ bị loại.

---

## 6. Kết luận & Định hướng

Enterprise Local RAG Stack v3.0 sở hữu một nền tảng kiến trúc **Advanced RAG** chuyên sâu. Thiết kế song song (async) trên các tác vụ Reformulation và Validation cho thấy hệ thống đã sẵn sàng để bứt phá về mặt hiệu năng khi được triển khai lên hạ tầng GPU Server, hoàn toàn có khả năng thay thế các dịch vụ Cloud RAG đắt đỏ, đảm bảo tính bảo mật và tính chính xác tuyệt đối.
