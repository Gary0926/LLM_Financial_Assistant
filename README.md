# LLM 財經助理
LLM 財經助理是一個大型語言模型，透過自然語言與使用者進行互動，即時回覆有關財經、投資等近期相關資訊。

## 功能
- 使用 LLM 和 LangChain 進行自然語言理解和生成。
- 串接 News API 獲取對話時所需相關文件。
- 使用 LangGraph 判定 LLM 和 API 的調用，以避免資源過度使用。
- 透過 RAG 技術使 LLM 能根據外部文件提供更即時具體的回答。

## 使用系統


## 呈現


## 操作步驟

## 金鑰需求
| 金鑰名稱 | 描述 | 備註 |
| ------- | ---- | ---- |
| OPENAI_API_KEY | 您的 OpenAI API 金鑰 | 必填 |
| LANGCHAIN_API_KEY | 您的 LANGCHAIN API 金鑰 | 選填 |
| NEWS_API_KEY | 您的 NEWS API 金鑰 | 必填 |
