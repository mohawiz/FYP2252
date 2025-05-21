import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import io
from PIL import Image
from io import BytesIO
import base64
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Set the page layout
st.set_page_config(layout="wide")

st.markdown("""
<div class="main-container">
    <h1>🧠 AI & Data Science Dashboard</h1>
    <p>Welcome to your personalized data science assistant!</p>
</div>
""", unsafe_allow_html=True)

cohere_api = os.getenv("COHERE_API_KEY")
groq_api = os.getenv("GROQ_API_KEY")

def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))

# Chat model
model = ChatGroq(temperature=0.3, groq_api_key=groq_api, model_name="meta-llama/llama-4-scout-17b-16e-instruct")
lida = Manager(text_gen=llm("cohere"))
textgen_config = TextGenerationConfig(n=1, temperature=0.3, model='command-r-08-2024', use_cache=True)

# Sidebar menu
st.sidebar.write("## 🔍 Explore")
menu = st.sidebar.selectbox("Choose an option", [
    '📄 Dataset Synopsis',
    '🎯 Goals',
    '🗂️🔎 Generate Graph from Data Query',
    '🧪 Data Profiling'
])

# 📄 Dataset Synopsis
if menu == "📄 Dataset Synopsis":
    st.header("📄 Dataset Synopsis")
    file_uploader = st.file_uploader("Upload your file", type='csv')

    if file_uploader is not None:
        path_to_save = 'file.csv'
        with open(path_to_save, 'wb') as f:
            f.write(file_uploader.getvalue())

        df = pd.read_csv(path_to_save)

        # LIDA Summary
        summary = lida.summarize('file.csv', summary_method='default', textgen_config=textgen_config)
        template = """
        Give the description about given data in 10 lines.

        Here is the json format summary of the data : {summary}

        Question: {question}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        user_input = "Make the description about the given data"
        result = chain.invoke({"question": user_input, "summary": summary})

        st.subheader("🧠 LLM-based Data Summary")
        st.write(result.content)
        st.write(summary)

# 🎯 Goals Section
elif menu == "🎯 Goals":
    st.header("🧭 Analytical Goals")
    file_uploader = st.file_uploader("Upload your file", type='csv')
    if file_uploader is not None:
        path_to_save = 'file.csv'
        with open(path_to_save, 'wb') as f:
            f.write(file_uploader.getvalue())

        summary = lida.summarize('file.csv', summary_method='default', textgen_config=textgen_config)
        num_goals = st.slider("Number of goals to generate", min_value=1, max_value=10, value=1)
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)

        st.write(f"## 🎯📌 Choose a Generated Goal ({len(goals)})")
        goal_questions = [goal.question for goal in goals]
        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)
        selected_goal_index = goal_questions.index(selected_goal)
        selected_goal_object = goals[selected_goal_index]
        st.write(goals[selected_goal_index])

        if selected_goal_object:
            st.write("📚 Plotting Libraries")
            visualization_libraries = ["seaborn", "matplotlib"]
            selected_library = st.selectbox('Choose a visualization library', options=visualization_libraries, index=0)

            st.write("📈 Graphical Insights")
            num_visualizations = st.slider("Number of visualizations to generate", min_value=1, max_value=5, value=1)
            textgen_config = TextGenerationConfig(n=num_visualizations, temperature=0.7, model='command-r-08-2024', use_cache=True)

            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            if visualizations:
                viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]
                selected_viz_title = st.selectbox('📌📉 Choose a Visualization', options=viz_titles, index=0)
                
                selected_viz = visualizations[viz_titles.index(selected_viz_title)]
                if selected_viz.raster:
                    imgdata = base64.b64decode(selected_viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    st.image(img, caption=selected_viz_title, use_column_width=True)

                st.write("🧠💻 Visualization Code")
                code = selected_viz.code
                st.code(code)

                st.write("🔍 Interpretation of Visuals")
                explain = lida.explain(code=code, library=selected_library, textgen_config=textgen_config)
                st.write(explain)
            else:
                st.warning("No visualizations were generated.")

# 🗂️🔎 Prompt-Based Graph Section
elif menu == "🗂️🔎 Generate Graph from Data Query":
    st.header("🗂️🔎 Generate Graph from Data Query")
    file_uploader = st.file_uploader("Upload your file", type='csv')
    if file_uploader is not None:
        path_to_save = 'file.csv'
        with open(path_to_save, 'wb') as f:
            f.write(file_uploader.getvalue())

    text_area = st.text_area("Enter a query about your data", height=200)
    st.write("## 📚 Visualization Library")
    visualization_libraries = ["seaborn", "matplotlib"]
    selected_library = st.selectbox('Choose a visualization library', options=visualization_libraries, index=0)

    if st.button("Generate Graph"):
        if len(text_area) > 0:
            st.info("Your Query: " + text_area)
            summary = lida.summarize('file.csv', summary_method='default', textgen_config=textgen_config)
            charts = lida.visualize(summary=summary, goal=text_area, library=selected_library, textgen_config=textgen_config)
            chart = charts[0]

            img = base64_to_image(chart.raster)
            st.image(img)

            st.write("### 🧠💻 Visualization Code")
            st.code(chart.code)

            st.write("### 🔍 Visualization Explanation")
            explain = lida.explain(code=chart.code, library=selected_library, textgen_config=textgen_config)
            st.write(explain)

# 🧪 Data Profiling Section
elif menu == "🧪 Data Profiling":
    st.header("🧪 Data Profiling")
    file_uploader = st.file_uploader("Upload your CSV file", type='csv')

    if file_uploader is not None:
        path_to_save = 'file.csv'
        with open(path_to_save, 'wb') as f:
            f.write(file_uploader.getvalue())

        df = pd.read_csv(path_to_save)

        st.subheader("📑 Automated YData Profiling Report")
        profile = ProfileReport(df, title="📊 Data Profiling Report", explorative=True)
        profile.to_file("report.html")

        with open("report.html", 'r', encoding='utf-8') as f:
            html = f.read()
            components.html(html, height=1000, scrolling=True)
