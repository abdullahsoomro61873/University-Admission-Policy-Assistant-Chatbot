import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask import Flask, request, render_template, Response
import warnings
warnings.filterwarnings("ignore")

# 1. Load Academic Policy Document
loader = TextLoader("policy.txt", encoding="utf-8")
documents = loader.load()

# 2. Split document into chunks (RAG)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# 4. Store in vector database
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 6. Load Ollama SLM
llm = OllamaLLM(model="phi3:mini", temperature=0, num_predict=400)

# 7. Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are the Mohammad Ali Jinnah University (MAJU) Admission Assistant. 
    Your role is to provide accurate, clear, and professional answers to student queries.

    GUIDELINES:
    1. PRIORITIZE: Use the provided Context as your primary source for specific details like fees, dates, and programs.
    2. AUGMENT: If the Context is missing specific details (like 'what is Biotechnology?'), use your internal knowledge to provide a concise explanation, but ensure it aligns with the MAJU context.
    3. NO "ROBOT-SPEAK": Never say "According to the document" or "The context states." Speak directly as the university's assistant.
    4. ACCURACY: If a specific date or fee is not in the context, do not guess. Instead, provide the general range if you know it, or advise the user to contact the admissions office for the latest 2026 updates.
    5. FORMATTING: Use bolding for key terms and bullet points for lists to make the answer scannable.
    6. TONE: Maintain a friendly, approachable, and professional tone throughout your responses.
    7. LIMITATIONS: If the question is unrelated to MAJU's academic policies or programs, politely inform the user that you can only assist with MAJU-related queries.
    8. CLARIFICATIONS: If the question is ambiguous, ask for clarification instead of making assumptions.
    9. complete your answers in maximum of 550 words.
    10. Always end your answers with "For more details, please visit the official MAJU website or contact the admissions office."
    11. CURRENT INFO: Ensure all information is up-to-date as of 2026.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# 8. RAG Chain (Modern LangChain)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



# ------------------- Flask Routes -------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return Response("Please ask a valid question.", mimetype='text/plain')
    
    def generate():
        for chunk in rag_chain.stream(question):
            yield chunk

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
