from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup
import os

app = Flask(__name__)
CORS(app)

# Set your GroqCloud API key
os.environ["GROQ_API_KEY"] = "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"

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n")

docs = []
index_text = extract_text_from_html("index.html")
docs.append(Document(page_content=index_text))

if os.path.exists("resume.html"):
    resume_text = extract_text_from_html("resume.html")
    docs.append(Document(page_content=resume_text))
else:
    resume_text = ""

# Optional: Save to .txt for backup
if not os.path.exists("docs"):
    os.makedirs("docs")
with open("docs/full_content.txt", "w", encoding="utf-8") as f:
    f.write(index_text + "\n" + resume_text)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

if not split_docs:
    raise ValueError("No text found in HTML files.")

# Use HuggingFace embeddings (local, no API needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()

# Use GroqCloud for LLM
llm = ChatGroq(
    groq_api_key="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",
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    max_tokens=256
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"response": "Please provide a message."})
        result = qa_chain.invoke({"query": user_message})
        return jsonify({"response": result["result"]})
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"response": f"Sorry, I encountered an error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
