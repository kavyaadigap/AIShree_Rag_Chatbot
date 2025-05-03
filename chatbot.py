import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Disable file watcher

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from PIL import Image
import base64

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 20px;
    }
    .block {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chat-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .example-box {
        background-color: #4A90E2;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .left-column {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .right-column {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 15px;
    }
    .text {
        font-size: 16px;
        line-height: 1.6;
        color: #333333;
    }
    .button {
        background-color: #4A90E2;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .button:hover {
        background-color: #357ABD;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<div class="title">AIShree Portfolio Chatbot</div>', unsafe_allow_html=True)

# Load your portfolio text
portfolio_file_path = r"C:\Users\Kavyasree P\OneDrive\Desktop\Learnings\DeepLearningPractice-master\Steamlit_learning\portfolio.txt"
with open(portfolio_file_path, "r", encoding="utf-8") as f:
    portfolio_text = f.read()

# Split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
texts = text_splitter.split_text(portfolio_text)

# Embeddings and Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(texts, embeddings)
vector_db.save_local("faiss_index")

vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def groq_response(question):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Create two columns for the two-color division
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="left-column">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Kavyashree P</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">**Email:** kavyaadigap@gmail.com  \n**Phone:** +918921701190  \n**LinkedIn:** [linkedin.com/kavyashree](https://linkedin.com/kavyashree)  \n**GitHub:** [github.com/kavyaadigap](https://github.com/kavyaadigap)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Skills</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">**Languages:** C, Python, SQL  \n**Technologies & Tools:** VS code, Jupyter Notebook, Power BI, AWS, GitHub, Scikit-learn, TensorFlow, Pandas, Numpy, Matplotlib, Data visualization, Data Analysis, Data Science, Machine Learning, Deep Learning, Artificial Intelligence, Predictive Modeling, Statistical Analysis, Supervised Learning, CAD, Adobe FrameMaker.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Work Experience</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">**Asset Integrity Engineering, Bangalore**  \n*June 2024 - Present*  \n**Data Scientist**  \n- Played a pivotal role in ensuring optimizing industrial process for clients responsible for 3% of the world’s oil production.  \n- Developed data driven solutions to monitor and predict asset performance leading 20% reduction in corrosion and maintenance costs.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="text">**Collins Aerospace, Bangalore**  \n*Aug 2022 - May 2024*  \n**Associate Engineer**  \n- Worked in the Mission Systems team supporting fuel components of Airbus and Boeing aircraft, focusing on testing, assembly/disassembly processes, and sensor data analysis.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Education</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">**MITE Karnataka**  \n*Aug 2018 - Jun 2022*  \n**B.E. in Aeronautical Engineering**  \n**CGPA:** 8.9/10</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Project Work</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">- **Predictive Maintenance of Aircraft Engine:** Designed and implemented a real-time aircraft engine performance monitoring system.  \n- **Weather monitoring IoT airship Using Machine Learning:** Engineered and deployed an IoT airship, applying expertise in aerodynamics and data analysis.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Awards and Certificates</div>', unsafe_allow_html=True)
    st.markdown('<div class="text">- Optimization of design using Machine learning  \n- AI Agents by Campus X  \n- Machine Learning and Data Science course from Bepec Solutions  \n- Cleared GATE exam in Data Science and Artificial Intelligence with AIR 1654</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right-column">', unsafe_allow_html=True)
    
    # Example Questions Block
    st.markdown('<div class="example-box">Here are some example questions you can ask:<br>- What are Kavya\'s skills?<br>- Can you tell me about Kavya\'s projects?<br>- What is Kavya\'s educational background?</div>', unsafe_allow_html=True)
    
    # Chat Box
    st.markdown('<div class="chat-box"><h3>Chat with AIShree</h3></div>', unsafe_allow_html=True)
    user_input = st.text_input("Your question:")

    if user_input:
        # Check for generic inputs
        if user_input.lower() in ["hi", "hello", "hey"]:
            response = "Hi! How are you? AIShree here, your AI assistant. How can I help you today?"
        else:
            # Retrieve relevant documents from the vector database
            documents = retriever.invoke(user_input)
            prompt = " ".join([doc.page_content for doc in documents])
            
            # Construct a conversational prompt for Groq
            groq_prompt =f"""You are AIShree, a friendly and helpful assistant designed exclusively to provide information about Kavya's professional portfolio.  
        The user asked: {user_input}  
        Here is some relevant information from Kavya's portfolio: {prompt}  

        Instructions for response:  
        - Limit your answer to 100 words.  
        - Only answer questions strictly related to Kavya's portfolio.  
        - If the user asks anything beyond her profile, politely respond:  
        "I'm AIShree, Kavya's portfolio chatbot. I can only answer questions related to her profile and cannot assist with topics beyond it."  
        - If the requested information is not available in the provided portfolio, reply:  
        "That information is not available in Kavya's profile."  
        - If the user repeatedly asks unrelated questions, gently remind them of the chatbot's purpose.  
        - Provide the information directly without referring to the portfolio or profile. Do not mention phrases like 'as mentioned in her profile' or 'from her portfolio.' Present the information as a fact.  
        - Maintain a polite, professional, and conversational tone at all times.  

        Please provide a helpful response accordingly.
        - When asked about Kavya's projects, list all relevant projects mentioned in the portfolio. Present them as a bulleted or numbered list for clarity.

    - If multiple projects are available, do not limit the response to just one—include all, ensuring each project description is concise but informative.
        """
            
            # Get response from Groq
            response = groq_response(groq_prompt)
        
        st.write(f"**Answer:** {response}")
    st.markdown('</div>', unsafe_allow_html=True)