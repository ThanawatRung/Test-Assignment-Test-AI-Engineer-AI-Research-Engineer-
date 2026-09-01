# Test-Assignment-Test-AI-Engineer-AI-Research-Engineer

โปรเจกต์นี้สร้างขึ้นเพื่อทำ test assignment ให้กับ BBL โดยเป็นระบบ **RAG (Retrieval-Augmented Generation) Agent** ที่ตอบคำถามเกี่ยวกับสุนัขพันธุ์ Corgi โดยใช้ [LangGraph](https://github.com/langchain-ai/langgraph) ในการ orchestrate การทำงานระหว่าง 2 เอเจนต์:

1. **Data Retriever Agent** — ค้นข้อมูลที่เกี่ยวข้องจาก Vector Database (Pinecone)
2. **Report Generator Agent** — สรุปข้อมูลที่ค้นได้ให้เป็นคำตอบฉบับสมบูรณ์

## โครงสร้างโปรเจกต์

```
.
├── main.py                                  # จุดเริ่มต้นโปรแกรม (chat loop)
├── Import_Knowledge_Data_to_VectorDB/
│   ├── knowledge.txt                        # ข้อมูลความรู้ดิบ (เกี่ยวกับ Corgi)
│   └── create_knowledge.py                  # สคริปต์นำข้อมูลเข้า Pinecone
├── Nodes/
│   ├── Data_Retreiver_RAG_Agent.py          # เอเจนต์ค้นข้อมูลจาก Pinecone
│   └── Report_Generator_Agent.py            # เอเจนต์สรุปรายงานคำตอบ
├── Orchestration/
│   ├── orchestration.py                     # ประกอบ workflow ด้วย LangGraph
│   └── state.py                             # นิยาม state ที่ใช้ส่งต่อระหว่าง node
├── Result_screenshots/                      # ตัวอย่างผลลัพธ์การรันจริง
└── requirements.txt
```

## การติดตั้ง (Installation)

1. สร้างและเปิดใช้งาน virtual environment (แนะนำ):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   ```

2. ติดตั้ง dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. สร้างไฟล์ `.env` ที่ root ของโปรเจกต์ (ดูรายละเอียดตัวแปรด้านล่าง)

## ไฟล์ `.env`

โปรเจกต์นี้อ่านค่า config ผ่าน `python-dotenv` (`load_dotenv()`) โดยต้องมีตัวแปรดังนี้:

- **`PINECONE_API_KEY`** — ใช้เชื่อมต่อ Pinecone เพื่อสร้าง/ค้น index ที่เก็บความรู้ (ใช้ทั้งตอน import ข้อมูลใน `create_knowledge.py` และตอนค้นข้อมูลใน `Data_Retreiver_RAG_Agent.py`) สมัครและสร้าง API key ได้ที่ [app.pinecone.io](https://app.pinecone.io) → **API Keys**

- **`OPENROUTER_API_KEY`** — ใช้เรียก LLM (`nvidia/nemotron-3.5-lightning:free`) ผ่าน OpenRouter สำหรับทั้งขั้นตอนค้นข้อมูล (tool-calling) และสรุปรายงาน สมัครและสร้าง API key ได้ที่ [openrouter.ai/keys](https://openrouter.ai/keys)

- **`HF_TOKEN`** — Hugging Face access token สร้างได้ที่ [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) ปัจจุบันมีไว้ใน `.env` แต่ยังไม่ถูกเรียกใช้ในโค้ดส่วนใดของโปรเจกต์

ตัวอย่างไฟล์ `.env`:

```env
PINECONE_API_KEY=your_pinecone_api_key
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

> **หมายเหตุด้านความปลอดภัย:** ไฟล์ `.env` ถูกใส่ไว้ใน `.gitignore` แล้ว จึงไม่ถูก commit ขึ้น git — อย่าลบออกจาก `.gitignore` และอย่า commit/แชร์ค่า API key จริงให้ผู้อื่นเห็นเด็ดขาด หากคีย์เคยหลุดออกไปแล้วให้ revoke แล้วสร้างใหม่ทันที

## วิธีใช้งาน

### ขั้นตอนที่ 1: Import Knowledge เข้า Pinecone (ต้องทำก่อนเสมอ)

ก่อนรัน `main.py` ต้องนำข้อมูลความรู้ (`Import_Knowledge_Data_to_VectorDB/knowledge.txt`) เข้า Pinecone index ก่อน ไม่เช่นนั้นเอเจนต์จะค้นข้อมูลไม่เจอ

```bash
python Import_Knowledge_Data_to_VectorDB/create_knowledge.py
```

สคริปต์นี้จะ:
- สร้าง Pinecone index ชื่อ `assignment-test-knowledge-bbl` อัตโนมัติ (ถ้ายังไม่มี) โดยใช้ embedding model `llama-text-embed-v2`
- แบ่งข้อความใน `knowledge.txt` เป็น chunk (chunk size 500, overlap 150)
- Upsert ข้อมูลเข้า namespace `knowledge` เป็นชุด (batch ละ 90 records)

รันคำสั่งนี้ทุกครั้งที่มีการอัปเดต/เพิ่มเนื้อหาใน `knowledge.txt` เพื่อ sync ข้อมูลใหม่เข้า Pinecone

### ขั้นตอนที่ 2: รันโปรแกรมหลัก

```bash
python main.py
```

เมื่อรันแล้วจะเข้าสู่โหมด chat ในเทอร์มินัล:

```
Chat with the agent. Type 'end' to quit.
You can ask questions about corgi dogs, and the agent will retrieve relevant information from the knowledge base and generate a report.

You: <พิมพ์คำถามเกี่ยวกับ Corgi ที่นี่>
```

- พิมพ์คำถามเกี่ยวกับสุนัขพันธุ์ Corgi แล้วกด Enter เอเจนต์จะค้นข้อมูลจาก Pinecone และสรุปคำตอบกลับมา
- พิมพ์ `end` เพื่อออกจากโปรแกรม

ตัวอย่างผลลัพธ์การรันจริงดูได้ที่โฟลเดอร์ [`Result_screenshots/`](Result_screenshots/)
