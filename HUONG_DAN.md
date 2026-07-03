# 📖 HƯỚNG DẪN CHI TIẾT (Tiếng Việt — dành cho người mới)

> File này giải thích **cặn kẽ từ A đến Z**: RAG là gì, project này làm gì, từng file
> có nhiệm vụ gì, cách cài đặt & chạy từng bước, và **kịch bản demo khi phỏng vấn**.

---

## 1. RAG là gì? (giải thích như cho người chưa biết gì)

Tưởng tượng ChatGPT là một người **rất thông minh nhưng trí nhớ đóng băng** tại thời
điểm nó được train. Hỏi nó về tài liệu nội bộ công ty bạn → nó không biết → nó **bịa**
(hallucination).

**RAG (Retrieval-Augmented Generation)** giải quyết bằng cách chia làm 2 bước:

1. **Retrieval (tìm kiếm)**: khi user hỏi, hệ thống đi **tìm những đoạn tài liệu liên
   quan nhất** trong kho tri thức (knowledge base) của bạn.
2. **Generation (sinh câu trả lời)**: đưa các đoạn tài liệu đó vào prompt, bảo LLM:
   *"Chỉ được trả lời dựa trên các đoạn này, kèm trích nguồn, không biết thì nói không biết."*

Kết quả: câu trả lời **chính xác, có nguồn, cập nhật được** mà không cần train lại model.
Đây là kiến trúc phổ biến nhất trong các hệ thống AI doanh nghiệp hiện nay — và là thứ
nhà tuyển dụng AI Engineer hỏi nhiều nhất.

## 2. Project này có gì "xịn" hơn RAG cơ bản?

RAG cơ bản (tutorial trên mạng) chỉ có: embed câu hỏi → tìm vector gần nhất → đưa vào LLM.
Project này làm **4 nâng cấp giống hệ thống production thật** (và khớp từng dòng trong CV):

| # | Kỹ thuật | Giải thích ngắn gọn |
|---|---|---|
| 1 | **Semantic chunking** | Cắt tài liệu theo **heading/đoạn văn** (ranh giới ngữ nghĩa) thay vì cắt cứng mỗi 500 ký tự → mỗi chunk nói về đúng 1 chủ đề → embedding "sắc nét" hơn. |
| 2 | **Hybrid search** | Chạy **2 kiểu tìm kiếm song song**: (a) *dense* — so sánh ý nghĩa bằng vector (FAISS), giỏi bắt câu diễn đạt khác từ nhưng cùng ý; (b) *sparse* — BM25 khớp từ khóa chính xác, giỏi bắt mã lỗi, tên riêng, thuật ngữ. Rồi **trộn kết quả bằng RRF** (Reciprocal Rank Fusion — cộng điểm 1/(60+hạng) từ mỗi bảng xếp hạng). |
| 3 | **Re-ranking** | Lấy ~20 ứng viên từ bước trên, cho qua **cross-encoder** (model đọc *cả câu hỏi lẫn đoạn văn cùng lúc* nên chấm điểm chính xác hơn nhiều) → chọn ra top 5 thật sự liên quan. |
| 4 | **Graceful fallback** | Nếu không có API key / LLM sập / hết tiền → hệ thống **không chết** mà trả về chế độ *extractive*: trích nguyên văn các đoạn liên quan nhất. Demo phỏng vấn không bao giờ đứng hình. |

Ngoài ra còn có **script đo lường (evaluation)** chứng minh bằng con số rằng
hybrid + re-ranking tốt hơn dense-only — cái này kể trong phỏng vấn cực kỳ ăn điểm.

## 3. Từng file làm gì?

```
RAG_project/
├── data/docs/            ← Kho tri thức: 8 file tài liệu tiếng Anh về AI/ML/RAG
│
├── src/rag/              ← TRÁI TIM của project
│   ├── config.py         ← Mọi cấu hình 1 chỗ: đường dẫn, tên model, top_k, đọc API key từ .env
│   ├── chunking.py       ← Cắt tài liệu thành chunk theo heading/đoạn văn (semantic chunking)
│   ├── embeddings.py     ← Biến text thành vector 384 chiều (sentence-transformers, chạy local, FREE)
│   ├── vectorstore.py    ← Lưu/tìm vector bằng FAISS (thư viện search vector của Meta)
│   ├── retrieval.py      ← ⭐ Hybrid search: FAISS + BM25 → trộn RRF → re-rank bằng cross-encoder
│   ├── generation.py     ← Gọi DeepSeek sinh câu trả lời có trích nguồn; lỗi thì fallback extractive
│   └── pipeline.py       ← Gom tất cả thành 1 hàm ask(question) → {answer, sources, timings}
│
├── scripts/
│   ├── build_index.py    ← Chạy 1 lần để "nạp kiến thức": docs → chunks → vectors → index/
│   └── evaluate.py       ← Đo hit-rate@3: dense-only vs hybrid vs hybrid+rerank
│
├── api/main.py           ← FastAPI server: POST /ask, GET /health, Swagger UI tại /docs
├── ui/app.py             ← Giao diện chat Streamlit (gọi API ở trên)
│
├── requirements.txt      ← Danh sách thư viện cần cài
├── .env.example          ← Mẫu file cấu hình key (copy thành .env rồi điền key)
└── .gitignore            ← Chặn không cho commit .env (bảo mật key!) và index/
```

**Luồng chạy khi user hỏi 1 câu:**

```
Câu hỏi → embed thành vector → FAISS tìm 20 chunk gần nghĩa nhất ─┐
        → tách từ khóa       → BM25 tìm 20 chunk khớp từ nhất  ───┤
                                                                   ├→ RRF trộn 2 bảng xếp hạng
                                                                   ├→ Cross-encoder chấm lại, lấy top 5
                                                                   ├→ Nhét 5 chunk vào prompt
                                                                   └→ DeepSeek trả lời + trích nguồn [1][2]
```

## 4. Cài đặt từng bước (Windows)

> Yêu cầu: Python 3.10+ ([python.org](https://python.org), nhớ tick "Add to PATH" khi cài).

Mở terminal (PowerShell) tại thư mục project rồi chạy lần lượt:

```powershell
# Bước 1: Tạo môi trường ảo (virtual environment — cái "hộp" chứa thư viện riêng cho project)
python -m venv .venv

# Bước 2: Kích hoạt môi trường ảo (dòng lệnh sẽ hiện (.venv) ở đầu)
.venv\Scripts\activate

# Bước 3: Cài thư viện (lần đầu tải ~500MB vì có PyTorch, chờ vài phút)
pip install -r requirements.txt

# Bước 4: Cấu hình API key (KHÔNG bắt buộc — không có key vẫn chạy được ở chế độ fallback)
copy .env.example .env
# → mở file .env bằng Notepad, thay your_deepseek_api_key_here bằng key thật
# → lấy key tại https://platform.deepseek.com (nạp $2 là dùng thoải mái)

# Bước 5: Build index — "nạp kiến thức" vào hệ thống (lần đầu tải model embedding ~90MB)
python scripts/build_index.py
# Thấy "Done in X.Xs. Index files written to ...\index" là OK
```

## 5. Chạy demo

Cần **2 cửa sổ terminal** (đều đã activate `.venv`):

```powershell
# Terminal 1 — chạy API server:
uvicorn api.main:app --port 8000
# Thấy "Application startup complete" là OK. Mở http://localhost:8000/docs xem Swagger UI.

# Terminal 2 — chạy giao diện chat:
streamlit run ui/app.py
# Trình duyệt tự mở http://localhost:8501
```

Gõ thử vào khung chat: `Why does hybrid search beat dense-only retrieval?`
→ nhận câu trả lời kèm nguồn `[1][2]`, bấm mở **📚 Sources** xem các chunk được
retrieve, chunk nào do `dense` tìm ra, chunk nào do `bm25`, và độ trễ từng bước.

Chạy đo lường (số liệu để kể khi phỏng vấn):

```powershell
python scripts/evaluate.py
```

Kết quả thực tế đo được trên corpus này:

| Cấu hình | Hit-rate@3 | MRR@3 |
|---|---|---|
| Dense-only (FAISS) | 93.8% | 0.938 |
| Hybrid (dense + BM25) | **100.0%** | **1.000** |
| Hybrid + re-ranking | **100.0%** | 0.969 |

→ Câu chuyện để kể: *"Hybrid cứu được câu hỏi mà dense-only trượt. Corpus demo nhỏ
(43 chunks) nên hybrid đã chạm trần 100%, re-ranker không còn chỗ để cộng thêm —
giá trị của re-ranker tăng dần theo kích thước corpus, khi kết quả tầng 1 nhiễu hơn.
Quan trọng là em có harness đo lường để chứng minh điều đó trên corpus thật."*

Độ trễ đo được (CPU, sau khi warm-up): retrieval ~9ms, BM25 <1ms,
re-rank ~380ms, DeepSeek generation ~1-3s.

## 6. 🎯 Kịch bản demo khi phỏng vấn (5 phút ăn điểm)

1. **Mở Streamlit**, hỏi 1 câu → chỉ vào câu trả lời có trích nguồn `[1][2]`:
   *"Mọi câu trả lời đều grounded — có thể audit ngược về đúng đoạn tài liệu."*
2. **Mở panel Sources** → chỉ vào cột `found by: dense / bm25`:
   *"Đây là hybrid search — em chạy song song semantic và lexical, trộn bằng RRF."*
3. **Tắt toggle "Cross-encoder re-ranking"** ở sidebar, hỏi lại cùng câu → so sánh nguồn
   thay đổi: *"Re-ranker quyết định chunk nào thật sự vào prompt."*
4. **Chạy `python scripts/evaluate.py`** → chỉ vào bảng số:
   *"Em đo hit-rate@3 bằng ablation — hybrid + rerank cao hơn dense-only rõ rệt."*
5. **Mở http://localhost:8000/docs** → *"Backend là FastAPI async, có Pydantic schema,
   health check, timing metrics từng stage — và nếu LLM provider sập thì degrade sang
   extractive answer chứ không trả lỗi 500."* (Có thể demo: đổi tên `DEEPSEEK_KEY`
   trong `.env` → restart API → hỏi lại → thấy badge "Extractive fallback".)

## 7. Câu hỏi phỏng vấn thường gặp & cách trả lời

**Q: Tại sao dùng hybrid mà không dùng dense-only?**
A: Embedding yếu với token lạ (mã sản phẩm, mã lỗi, viết tắt) vì model chưa từng học;
BM25 yếu khi user diễn đạt khác từ ngữ trong tài liệu. Chạy cả hai rồi fuse bằng RRF
lấy được điểm mạnh của cả hai — benchmark doanh nghiệp thường tăng recall 10–25%.

**Q: RRF là gì, sao không cộng thẳng điểm?**
A: Điểm BM25 (0→vô cùng) và cosine (-1→1) khác thang đo, cộng thẳng phải normalize và
tune trọng số. RRF chỉ dùng **thứ hạng**: score = Σ 1/(60 + rank) — miễn nhiễm khác
thang đo, không cần tune, là chuẩn de-facto trong production.

**Q: Bi-encoder vs cross-encoder khác gì?**
A: Bi-encoder embed câu hỏi và tài liệu **riêng rẽ** rồi so cosine — nhanh (precompute
được) nhưng hai bên không "nhìn thấy nhau". Cross-encoder ghép cả hai vào một input,
mọi token attend lẫn nhau — chính xác hơn hẳn nhưng chậm, nên chỉ dùng re-rank ~20
ứng viên cuối.

**Q: Làm sao đo chất lượng RAG?**
A: Tách 2 tầng: retrieval đo hit-rate@k / MRR trên bộ câu hỏi có gán nguồn đúng;
generation đo faithfulness / answer relevance bằng LLM-as-judge (RAGAS). Project này
có `scripts/evaluate.py` làm ablation retrieval.

**Q: Đưa lên production thì thêm gì?**
A: Thay FAISS bằng vector DB có filtering + replication (Qdrant/Pinecone); thêm
caching (embedding, answer); auth + rate limiting; Docker + K8s + CI/CD; observability
(trace từng stage, cost tracking); index rebuild theo blue-green; chống prompt
injection từ tài liệu.

**Q: Sao dùng DeepSeek?**
A: API tương thích chuẩn OpenAI (đổi provider chỉ 1 dòng config), rẻ, chất lượng tốt.
Kiến trúc không khóa vào provider nào — swap sang Claude/GPT chỉ cần sửa `config.py`.

## 8. Lỗi thường gặp

| Lỗi | Nguyên nhân & cách sửa |
|---|---|
| `Index not found... Run build_index.py first` | Chưa build index → chạy `python scripts/build_index.py` |
| `ModuleNotFoundError: No module named 'rag'` | Chạy script từ sai thư mục → phải đứng ở thư mục gốc project |
| Streamlit báo "API offline" | Chưa chạy Terminal 1 (`uvicorn api.main:app --port 8000`) |
| Câu trả lời hiện "Extractive fallback" | Không có/sai `DEEPSEEK_KEY` trong `.env` — vẫn là feature, không phải bug 😄 |
| pip cài chậm/lỗi | Thử `pip install -r requirements.txt --timeout 120`; cần mạng ổn định vì tải PyTorch |

---

Chúc ông demo thành công và đậu phỏng vấn! 🚀
