import streamlit as st
import cohere
from pinecone import Pinecone
from FlagEmbedding import BGEM3FlagModel
from google import genai

# 1. Page Config
st.set_page_config(page_title="YJIS | المستشار القضائي اليمني", page_icon="⚖️", layout="wide")

# 2. Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ نظام المعلومات القضائية اليمني (YJIS)")
st.subheader("إصدار النشر الدائم v3.5.1")

# 3. Secure Key Management (Using Streamlit Secrets)
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    INDEX_NAME = "yemeni-laws-index"
except Exception:
    st.error("⚠️ خطأ: مفاتيح API غير موجودة في إعدادات Secrets.")
    st.info("يرجى إضافة المفاتيح في لوحة تحكم Streamlit Cloud -> Settings -> Secrets")
    st.stop()

# 4. Model Loading (Cached)
@st.cache_resource(show_spinner="جاري تحميل محرك الذكاء الاصطناعي... قد يستغرق دقيقة في المرة الأولى")
def load_models():
    # Use CPU-only to save RAM on hosting servers
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

model = load_models()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
co = cohere.Client(COHERE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def convert_query_sparse(bge_sparse_dict):
    unique_tokens = {}
    for token, weight in bge_sparse_dict.items():
        token_id = model.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None:
            unique_tokens[token_id] = max(unique_tokens.get(token_id, 0), float(weight))
    return {"indices": list(unique_tokens.keys()), "values": list(unique_tokens.values())}

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("اسأل عن أي مادة قانونية يمنية..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري استخراج وتحليل النصوص القانونية..."):
            # Retrieval
            embeddings = model.encode([prompt], return_dense=True, return_sparse=True)
            dense_vec = embeddings['dense_vecs'][0].tolist()
            sparse_vec = convert_query_sparse(embeddings['lexical_weights'][0])
            
            res = index.query(vector=dense_vec, sparse_vector=sparse_vec, top_k=15, include_metadata=True)
            
            if res['matches']:
                docs = [m['metadata']['text'] for m in res['matches']]
                law_titles = [m['metadata'].get('law_title', 'قانون يمني') for m in res['matches']]
                
                # Rerank
                rerank = co.rerank(model="rerank-multilingual-v3.0", query=prompt, documents=docs, top_n=3)
                
                context = ""
                for r in rerank.results:
                    context += f"[المصدر: {law_titles[r.index]}]\n{docs[r.index]}\n\n"
                
                # Generate
                sys_prompt = f"أنت مستشار قانوني يمني. أجب بدقة بناءً على النصوص التالية فقط:\n\n{context}\n\nالسؤال: {prompt}"
                response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=sys_prompt)
                
                answer = response.text
                st.markdown(answer)
                with st.expander("المراجع القانونية"):
                    st.write(context)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.write("لم أجد نصوصاً مطابقة.")
