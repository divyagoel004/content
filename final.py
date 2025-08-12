import streamlit as st
import json
from autogen import AssistantAgent, config_list_from_json
from base64 import b64decode
import requests
import torch
from io import BytesIO
import numpy as np
import os
import re

from agent import search_and_embed
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter
from PIL import Image
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from time import sleep
from collections import defaultdict
import ast
from cleanup_images import start_cleanup_daemon

start_cleanup_daemon()
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import re

# New imports for DuckDuckGo search and vector database
# from duckduckgo_search import DDGS
# import chromadb
# from chromadb.utils import embedding_functions
import hashlib
from urllib.parse import urljoin, urlparse
import trafilatura
import asyncio
import aiohttp
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
import uuid
from base import serper_search , query_vector_db
# Define the structure to parse
class CodeSnippet(BaseModel):
    code: str = Field(..., description="Concise Python code snippet")
parser = PydanticOutputParser(pydantic_object=CodeSnippet)

# API Keys (only Gemini needed now)
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize vector database
# @st.cache_resource
# def initialize_vector_db():
#     """Initialize ChromaDB client and collection"""
#     client = chromadb.PersistentClient(path="./chroma_db")
    
#     # Create embedding function
#     sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
    
#     collection = client.get_or_create_collection(
#         name="presentation_content",
#         embedding_function=sentence_transformer_ef
#     )
    
#     return client, collection

# # Initialize sentence transformer for content processing
# @st.cache_resource
# def load_sentence_transformer():
#     return SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit page config for wide layout
st.set_page_config(layout="wide", page_title="AI Presentation Studio", page_icon="🎯")
from langfuse import Langfuse

# Initialize Langfuse client
langfuse = Langfuse(
    public_key="pk-lf-ff2508c5-f759-49c2-b1ef-b8c42c3cdd4a",
    secret_key="sk-lf-29906660-5e8e-4ba1-ab9b-147fa5b270ff",
    host="https://cloud.langfuse.com",
)






st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container styling */
.main-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    padding: 20px;
}

/* Slide container */
.slide-container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin: 20px 0;
    overflow: hidden;
    transition: all 0.3s ease;
    position: relative;
}

.slide-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.15);
}

/* Slide header */
.slide-header {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    
    font-size: 20px !important;   
    padding: 20px 25px !important;
    font-weight: 700;
    position: relative;
}

.slide-number {
    position: absolute;
    top: 15px;
    right: 25px;
    background: rgba(255,255,255,0.2);
    padding: 5px 12px;
    border-radius: 15px;
    font-size: 14px;
}

/* Slide content */
.slide-content {
    padding: 30px;
    min-height: 400px;
    font-size: 16px;
    line-height: 1.6;
}

/* Interactive elements */
.editable-section {
    border: 2px dashed transparent;
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
}

.editable-section:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
}

.edit-button {
    position: absolute;
    top: 5px;
    right: 5px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 15px;
    padding: 5px 10px;
    font-size: 12px;
    cursor: pointer;
    opacity: 0;
    transition: all 0.3s ease;
}

.editable-section:hover .edit-button {
    opacity: 1;
}

/* Sidebar styling */
.sidebar-content {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

/* Control buttons */
.control-button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 25px;
    margin: 5px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}

.control-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Navigation */
.slide-navigation {
    position: fixed;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-radius: 25px;
    padding: 10px 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    z-index: 1000;
}

/* Progress bar */
.progress-bar {
    height: 4px;
    background: linear-gradient(45deg, #667eea, #764ba2);
    border-radius: 2px;
    margin: 20px 0;
    transition: width 0.3s ease;
}

/* Animation classes */
.fade-in {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Comment box */
.comment-box {
    background: #f8f9ff;
    border-left: 4px solid #667eea;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 10px 10px 0;
}

/* Research indicator */
.research-badge {
    background: linear-gradient(45deg, #11998e, #38ef7d);
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 10px;
}

/* Loading animation */
.loading-spinner {
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'slides' not in st.session_state:
    st.session_state.slides = []
if 'current_slide' not in st.session_state:
    st.session_state.current_slide = 0
if 'presentation_mode' not in st.session_state:
    st.session_state.presentation_mode = False
if 'editing_section' not in st.session_state:
    st.session_state.editing_section = None
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = {}
if 'comments' not in st.session_state:
    st.session_state.comments = {}

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize vector database
# vector_client, collection = initialize_vector_db()
# sentence_model = load_sentence_transformer()


def generate_mermaid_diagram(payload: dict, vm_ip: str = "40.81.228.142:5500") -> str:
    url = f"http://{vm_ip}/render-mermaid/"
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

# Load configuration files (you'll need to adjust these paths)
try:
    config_list = config_list_from_json(env_or_file="config.txt")
    llm_config = {
        "seed": 44,
        "config_list": config_list,
        "temperature": 0
    }

    with open("file.json", "r") as f:
        content_types_dict = json.load(f)

    with open("component.txt", "r") as f:
        available_components = [line.strip() for line in f]
except FileNotFoundError as e:
    st.error(f"Configuration file not found: {e}")
    st.stop()

def get_top_image_contexts(topic, content_type, component, top_k=10):
    """
    Retrieve top-k image paths based on similarity between text and image embeddings.
    Enhanced with vector database integration.
    """
    
    # Choose correct embedding and path file based on content_type        
    if component.lower() == "Real-World Photo":
        emb_file = "embeddings_dino/real_images.npy"
        path_file = "embeddings_dino/real_images_paths.txt"
        query = f"Extract the relevant data for generating {component} on {content_type} of {topic}"
    else:
        emb_file = "embeddings_dino/diagrams.npy"
        path_file = "embeddings_dino/diagrams_paths.txt"
        query = f"Extract the relevant data on {content_type} of {topic} for generating {component}"

    # Load image embeddings and image paths
    image_emb = np.load(emb_file)
    with open(path_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    if len(image_paths) != len(image_emb):
        raise ValueError("Mismatch between image paths and embedding vectors.")

    # Load sentence transformer for encoding the query
    text_encoder = SentenceTransformer("all-mpnet-base-v2")
    query_emb = text_encoder.encode(query, normalize_embeddings=True)

    # Convert to tensors
    query_emb = torch.tensor(query_emb, dtype=torch.float32)
    image_emb = torch.tensor(image_emb, dtype=torch.float32)

    # Compute cosine similarity
    scores = torch.nn.functional.cosine_similarity(query_emb, image_emb)
    top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices.numpy()

    # Fetch the corresponding top image paths
    top_contexts = [image_paths[i] for i in top_indices]
    return "\n".join(top_contexts)

# Updated AutoGen agents with vector database integration
knowledge_base_agent = AssistantAgent(
    name="Knowledge-Base-Creator",
    system_message='''You are a professional research agent that creates comprehensive knowledge bases from vector database search results.

Your task is to:
1. Analyze content retrieved from vector database containing papers, ebooks, blogs, documentation, and articles
2. Extract key concepts, definitions, examples, and insights from the full content
3. Organize information into structured knowledge base
4. Identify gaps where additional research might be needed
5. Provide quality scores for different sources based on content depth and relevance

Instructions:
- Create a structured summary with key sections: Definitions, Core Concepts, Applications, Examples, Latest Trends
- Rate source credibility (1-10) based on content quality and source authority
- Highlight conflicting information if found
- Suggest additional search terms if gaps are identified
- Output should be in JSON format for easy processing
- Use the full content provided, not just snippets

Output Format:
{
  "knowledge_base": {
    "definitions": ["..."],
    "core_concepts": ["..."],
    "applications": ["..."],
    "examples": ["..."],
    "latest_trends": ["..."]
  },
  "source_quality": {
    "high_quality_sources": ["..."],
    "medium_quality_sources": ["..."],
    "gaps_identified": ["..."]
  },
  "additional_search_terms": ["..."]
}
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

content_enrichment_agent = AssistantAgent(
    name="Content-Enrichment-Agent",
    system_message='''You are a professional content enrichment specialist with access to a comprehensive vector database.

Your task is to enhance slide content using the researched knowledge base and validated resources from the vector database.

Instructions:
- Use knowledge base extracted from vector database to add depth and accuracy to content
- Incorporate latest trends and real-world examples from the full content retrieved
- Ensure technical accuracy using verified sources from academic papers and documentation
- Add citations and references where appropriate using the source URLs provided
- Maintain professional presentation style
- Balance comprehensiveness with slide readability
- Leverage the full content available in the vector database, not just snippets

You will receive:
1. Original content request
2. Knowledge base with researched information from vector database
3. Validated images and visual concepts
4. Topic metadata (depth level, audience, etc.)
5. Full content from relevant documents

Output should be enhanced, accurate, and citation-rich content ready for slides.
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

content_type_Selector = AssistantAgent(
    name="Content-Type-Selector",
    system_message='''You are a professional slide content type selection agent for PowerPoint with access to comprehensive research data.

Instructions:
- You will be given a topic, knowledge base extracted from vector database, and a JSON dictionary of available content types.
- Use the comprehensive knowledge base to make informed decisions about content types
- Consider the depth and variety of content available in the vector database
- Select only the 4 to 6 most relevant content types based on the topic and available research.
- Output must be a single line of comma-separated values with **no explanations**, **no formatting**, and **no extra text**.

❗Only output the selected content types as:
ContentType1, ContentType2, ContentType3, ContentType4

Do NOT include:
- Any prefix or introduction.
- Any markdown or formatting.
- Any additional commentary.
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

component_type_Selector = AssistantAgent(
    name="Component-Type-Selector",
    system_message='''You are a professional slide component selector for each content type with access to comprehensive research data.

Your job is to select 4 to 6 components for each content-type, based on:
- The topic
- The researched knowledge base extracted from vector database
- The default mapping provided in a txt file (as a Python dictionary)
- Available visual resources and validated images
- Full content depth available in the vector database

Instructions:
- Use the comprehensive knowledge base to make informed component selections
- Use the default values as a starting point, and edit them based on research findings from vector database
- Consider available images and visual concepts when selecting components
- Leverage the full content available, not just snippets
- You must return a Python dictionary, where keys are content-types and values are Python lists of selected components.
- Only output the final dictionary. Do not include any reasoning, explanations, or extra text.

✅ Output Format Example (nothing else):
{
  "Content-type1": ["Component A", "Component B", "Component C"],
  "Content-type2": ["Component X", "Component Y"]
}

❌ Do NOT include:
- Any markdown
- Any text before or after the dictionary
- Any reasoning or explanation
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

content_generator = AssistantAgent(
    name="Content-generator",
    system_message='''You are a professional content generator for presentations with access to comprehensive research from vector database.

You are an expert at generating one of the following content types at a time:
- Text
- Code Snippet
- Mathematical Equation
- Table

Instructions:
- You will be provided with researched knowledge base and validated sources from vector database
- Use this comprehensive information to create accurate, up-to-date, and comprehensive content
- Include relevant examples and latest trends from the research
- Leverage full content from academic papers, documentation, and expert sources
- Only generate the requested type. Do not include anything else.
- Do not provide explanations or extra content.
- Your output must be in a format suitable for direct use in a slide.
- When generating text, incorporate findings from the comprehensive knowledge base
- For code snippets, use best practices and latest standards found in research
- For equations, ensure mathematical accuracy using verified sources
- Include citations when using specific information from sources

❌ Do NOT include:
- Headings or labels (like "Here is your text")
- Multiple content types at once
- Reasoning or notes

✅ Only respond with the requested content enhanced by comprehensive research findings.
''',
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Slide class for better organization
class Slide:
    def __init__(self, title, content_sections, slide_number):
        self.title = title
        self.content_sections = content_sections  # List of {"type": "text/image/code", "content": "...", "editable": True}
        self.slide_number = slide_number
        self.comments = []
        self.research_sources = []




def clean_latex_equation(math_expr: str) -> str:
    """
    Clean and normalize LaTeX equation string
    """
    # Remove common wrapper symbols
    expr = math_expr.strip()
    expr = re.sub(r'^\$+|\$+$', '', expr)  # Remove leading/trailing $
    expr = re.sub(r'^\\begin{equation}|\\end{equation}$', '', expr)  # Remove equation environment
    expr = re.sub(r'^\\begin{align}|\\end{align}$', '', expr)  # Remove align environment
    
    # Fix common LaTeX issues
    expr = re.sub(r'\\text{([^}]*)}', r'\\mathrm{\1}', expr)  # Replace \text with \mathrm
    expr = re.sub(r'\\textrm{([^}]*)}', r'\\mathrm{\1}', expr)  # Replace \textrm with \mathrm
    
    # Clean up whitespace
    expr = re.sub(r'\s+', ' ', expr).strip()
    
    return expr

def render_latex_to_image(math_expr: str, output_path="equation.png") -> str:
    """
    Render a LaTeX equation as a high-quality PNG image
    
    Args:
        math_expr: LaTeX equation string (without $ delimiters)
        output_path: Output file path
        
    Returns:
        Absolute path to the generated image
    """
    try:
        # Clean the equation
        expr = clean_latex_equation(math_expr)
        
        # Create figure with high DPI for crisp rendering
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
        ax.axis('off')
        
        # Render the equation
        ax.text(
            0.5, 0.5,
            f"${expr}$",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=24,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        
        # Set tight layout
        plt.tight_layout(pad=0.2)
        
        # Save with high quality
        abs_path = os.path.abspath(output_path)
        fig.savefig(
            abs_path, 
            bbox_inches='tight', 
            transparent=False,  # White background for better visibility
            dpi=300,
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        return abs_path
        
    except Exception as e:
        print(f"ERROR: Failed to render LaTeX equation: {str(e)}")
        # Create a simple text fallback image
        return create_fallback_equation_image(math_expr, output_path)

def create_fallback_equation_image(math_expr: str, output_path: str) -> str:
    """
    Create a simple text-based fallback when LaTeX rendering fails
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
        ax.axis('off')
        
        # Display as plain text
        ax.text(
            0.5, 0.5,
            f"Equation: {math_expr}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8)
        )
        
        plt.tight_layout()
        abs_path = os.path.abspath(output_path)
        fig.savefig(abs_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        
        return abs_path
        
    except Exception as e:
        print(f"ERROR: Even fallback image creation failed: {str(e)}")
        return ""
    


def generate_presentation_slides(topic, depth_level="intermediate"):
    """Generate slides with research-enhanced content from vector database"""
    
    # Initialize Langfuse
     # Assuming langfuse is already configured
    
    # Create main trace for presentation generation
    import time
    trace = langfuse.trace(
        name="generate_presentation_slides",
        input={
            "topic": topic,
            "depth_level": depth_level
        },
        metadata={
            "function": "generate_presentation_slides",
            "timestamp": time.time()
        }
    )
    
    # Create knowledge base
    serper_search_span = langfuse.span(
        trace_id=trace.id,
        name="serper_search",
        input={"topic": topic, "count": 20}
    )
    serper_search(topic, 20)
    serper_search_span.end()
    
    # Content type selection
    prompt = f"Select the appropriate content-types from {list(content_types_dict.keys())} for the topic \"{topic}\""
    
    # Log the prompt
    langfuse.create_prompt(
        name="content_type_selection_prompt",
        prompt=prompt,
        type="text"
    )
    
    content_type_selection_span = langfuse.span(
        trace_id=trace.id,
        name="content_type_selection",
        input={"prompt": prompt}
    )
    
    output1 = content_type_Selector.generate_reply(messages=[{"role": "user", "content": prompt}])
    selected_content_types = [c.strip() for c in output1.split(",")]
    
    content_type_selection_span.update(
        output={
            "raw_output": output1,
            "selected_content_types": selected_content_types
        }
    )
    content_type_selection_span.end()
    
    # Prepare component dictionary with research context
    v = {k: content_types_dict[k] for k in selected_content_types if k in content_types_dict}

    prompt2 = f'''For the content-types {output1}, the default selected components are: {v}.
    Based on the full component list {available_components}, suggest edits.'''
    
    # Log the prompt
    langfuse.create_prompt(
        name="component_selection_prompt", 
        prompt=prompt2,
        type="text"
    )
    
    component_selection_span = langfuse.span(
        trace_id=trace.id,
        name="component_selection",
        input={"prompt": prompt2, "default_components": v}
    )
    
    component_dict = component_type_Selector.generate_reply(messages=[{"role": "user", "content": prompt2}])
    
    # Search and embed
    search_embed_span = langfuse.span(
        trace_id=trace.id,
        name="search_and_embed",
        input={"topic": topic, "max_images": 100}
    )
    search_and_embed(f"{topic}", max_images=100)
    search_embed_span.end()
    
    try:
        component_dict = eval(component_dict)
        component_selection_span.update(
            output={"component_dict": component_dict, "parsing_success": True}
        )
    except:
        component_dict = v  # Fallback to default
        component_selection_span.update(
            output={"component_dict": component_dict, "parsing_success": False, "fallback_used": True}
        )
    
    component_selection_span.end()
    
    slides = []
    slide_number = 1
    
    # Generate slides - Track each slide generation
    print(component_dict)
    for content_type, components in component_dict.items():
        
        # Create slide-level span
        slide_span = langfuse.span(
            trace_id=trace.id,
            name=f"generate_slide_{slide_number}",
            input={
                "content_type": content_type,
                "components": components,
                "slide_number": slide_number
            }
        )
        
        # Generate slide title with knowledge base query
        query = f"generate the context for {content_type} of {topic}"
        
        # Track knowledge base query
        kb_query_span = langfuse.span(
            trace_id=trace.id,
            name="knowledge_base_query",
            input={"query": query}
        )
        
        knowledge_base = query_vector_db(query)
        st.session_state.knowledge_base = knowledge_base
        
        kb_query_span.update(
            output={
                "knowledge_base_content": knowledge_base,
                "knowledge_base_length": len(str(knowledge_base)) if knowledge_base else 0
            }
        )
        kb_query_span.end()
        
        # Generate slide title
        prompt1 = f"""
                Generate a professional slide title for topic '{topic}' focused on '{content_type}'.
                
                Use this research context:
                {knowledge_base}
                
                Title should be:
                - Professional and engaging
                - Informed by research findings
                - Suitable for {depth_level} level audience
                
                Return only the title.
                """
        
        # Log the prompt
        langfuse.create_prompt(
            name="slide_title_generation_prompt",
            prompt=prompt1,
            type="text"
        )
        
        title_generation_span = langfuse.span(
            trace_id=trace.id,
            name="slide_title_generation",
            input={
                "prompt": prompt1,
                "knowledge_base_used": bool(knowledge_base)
            }
        )
        
        slide_title = content_enrichment_agent.generate_reply([
            {
                "role": "user",
                "content": prompt1
            }
        ])
        
        title_generation_span.update(output={"slide_title": slide_title})
        title_generation_span.end()
        
        # Generate content sections
        content_sections = []
        
        for component_idx, component in enumerate(components):
            # Create component-level span
            
            component_span = langfuse.span(
                trace_id=trace.id,
                name=f"generate_component_{component_idx}",
                input={
                    "component": component,
                    "component_index": component_idx
                }
            )
            
            research_context = f"""
            Knowledge Base: {knowledge_base}
            Topic: {topic}
            Content Type: {content_type}
            Component: {component}
            Depth Level: {depth_level}
            """
            
            component_span.update(input={"research_context": research_context})
            
            if component.lower() == "text" and content_type.lower() == "definition":
                p = f"""
                        Use this research context only as a reference:

                            {research_context}

                            Create clean, slide-ready content.

                            Instructions:
                            - Start with a short **Definition** of **{topic}** (max 2 lines, aim for 1–1.5)
                            - Add 4–5 crisp **Key Points** (each ≤ 1.5 lines, ideally 1)
                            - List 3–4 precise **Real-World Applications** (≤ 1.5 lines each)
                            - Use external knowledge, not just input
                            - Avoid fluff, rewording, or long phrases
                            - Tone: clear, minimal, and professional ({depth_level} audience)
                            - Format: {component} → {content_type}
                        """
                
                # Log the prompt
                langfuse.create_prompt(
                    name="text_definition_generation_prompt",
                    prompt=p,
                    type="text"
                )
                
                text_generation_span = langfuse.span(
                    trace_id=trace.id,
                    name="text_definition_generation",
                    input={"prompt": p}
                )
                
                text_content = content_generator.generate_reply([
                    {
                         "role": "user",
                        "content": p
                    }
                ])
                
                text_generation_span.update(output={"text_content": text_content})
                text_generation_span.end()
                
                content_sections.append({
                    "type": "text",
                    "content": text_content.strip(),
                    "editable": True,
                    "component_name": component
                })
                import re
            elif component.lower() == "text" and content_type.lower() != "definition":
                p = f"""
                        Use this research context only as a reference:

                        {research_context}

                        Generate brief, original slide content.

                        Instructions:
                        - Begin with a short **headline** for the {content_type}
                        - Add 4–5 **Key Points** (≤ 1.5 lines each, aim for 1 line)
                        - Add 2–3 **Real-World Applications** (1–1.5 lines each)
                        - Only use external knowledge
                        - Keep tone professional and minimal ({depth_level} audience)
                        - Format strictly: {component} → {content_type}
                        - ⚠️ No rewording of input — use fresh phrasing
                        """
                
                # Log the prompt
                langfuse.create_prompt(
                    name="text_general_generation_prompt",
                    prompt=p,
                    type="text"
                )
                
                text_generation_span = langfuse.span(
                    trace_id=trace.id,
                    name="text_general_generation",
                    input={"prompt": p}
                )
                
                text_content = content_generator.generate_reply([
                    {
                         "role": "user",
                        "content": p
                    }
                ])
                
                text_generation_span.update(output={"text_content": text_content})
                text_generation_span.end()
                
                content_sections.append({
                    "type": "text",
                    "content": text_content.strip(),
                    "editable": True,
                    "component_name": component
                })
            
            elif component.lower() == "code snippet":
                heading = f"**{component}**"
                
                content_sections.append({
                    "type": "text",
                    "content": heading,
                    "editable": False,
                    "component_name": "heading"
                })
                
                template = PromptTemplate(
                    template="""
                Generate a concise Python code snippet for the topic: '{topic}'.

                Requirements:
                - Fully runnable (no syntax errors)
                - ≤ 30 lines
                - 4-space indentation
                - Use functions/classes if useful
                - Avoid markdown formatting like backticks

                {format_instructions}
                """,
                    input_variables=["topic"],
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )

                prompt = template.format(topic=topic)
                
                # Log the prompt
                langfuse.create_prompt(
                    name="code_snippet_generation_prompt",
                    prompt=prompt,
                    type="text"
                )
                
                code_generation_span = langfuse.span(
                    trace_id=trace.id,
                    name="code_snippet_generation",
                    input={"prompt": prompt}
                )
                
                raw_output = content_generator.generate_reply([{
                    "role": "user",
                    "content": prompt
                }])

                # Try structured parsing
                try:
                    parsed = parser.parse(raw_output)
                    code_clean = parsed.code
                    parsing_success = True
                except Exception as parse_error:
                    # Fallback to regex extraction
                    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
                    code_clean = match.group(1).strip() if match else raw_output.strip()
                    parsing_success = False
                
                code_generation_span.update(
                    output={
                        "raw_output": raw_output,
                        "code_clean": code_clean,
                        "parsing_success": parsing_success
                    }
                )
                code_generation_span.end()

                content_sections.append({
                    "type": "code",
                    "content": code_clean,
                    "editable": True,
                    "component_name": component
                })
                
            elif component.lower() == "mathematical equations":
                import os
                heading = f"**{component}**"
                
                content_sections.append({
                    "type": "text",
                    "content": heading,
                    "editable": False,
                    "component_name": "heading"
                })
                
                try:
                    pro = f"""
                        You are a mathematics educator preparing a slide for a technical presentation.
                        Generate a professional and relevant mathematical equation or formula related to the topic: '{topic}'.
                        
                        Requirements:
                        - Provide ONE clear, well-formatted LaTeX equation
                        - Include a brief 1-2 sentence description of what the equation represents
                        - Use proper LaTeX syntax (avoid \\text, use \\mathrm instead)
                        - Do not wrap the equation in $$ symbols
                        - Keep the equation concise and readable
                        
                        Format your response as:
                        EQUATION: [your LaTeX equation here]
                        DESCRIPTION: [brief description]
                        """
                    
                    # Log the prompt
                    langfuse.create_prompt(
                        name="mathematical_equation_generation_prompt",
                        prompt=pro,
                        type="text"
                    )
                    
                    math_generation_span = langfuse.span(
                        trace_id=trace.id,
                        name="mathematical_equation_generation",
                        input={"prompt": pro}
                    )
                    
                    out = content_generator.generate_reply([
                        {
                            "role": "user",
                            "content": pro
                        }
                    ])
                    
                    # Parse the response
                    lines = out.strip().split('\n')
                    latex_equation = ""
                    description = ""
                    
                    for line in lines:
                        if line.startswith("EQUATION:"):
                            latex_equation = line.replace("EQUATION:", "").strip()
                        elif line.startswith("DESCRIPTION:"):
                            description = line.replace("DESCRIPTION:", "").strip()
                    
                    # Fallback if parsing fails
                    if not latex_equation:
                        latex_equation = out.strip()
                    
                    # Clean the LaTeX equation
                    latex_equation = clean_latex_equation(latex_equation)
                    
                    math_generation_span.update(
                        output={
                            "raw_output": out,
                            "latex_equation": latex_equation,
                            # "description": description
                        }
                    )
                    
                    # Generate image
                    img_path = render_latex_to_image(
                        latex_equation,
                        output_path=f"{component.lower().replace(' ', '_')}_latex.png"
                    )
                    
                    if os.path.exists(img_path):
                        print(f"DEBUG: Successfully saved equation image: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                        
                        math_generation_span.update(
                            output={
                                "image_path": img_path,
                                "image_size": os.path.getsize(img_path),
                                "generation_success": True
                            }
                        )
                        
                        # Add the equation image
                        content_sections.append({
                            "type": "image",
                            "content": img_path,
                            "editable": True,
                            "component_name": component
                        })
                        
                        # Add description if available
                        if description:
                            content_sections.append({
                                "type": "text",
                                "content": description,
                                "editable": True,
                                "component_name": f"{component}_description"
                            })
                    else:
                        raise Exception("Equation image file was not created successfully")
                    
                    math_generation_span.end()
                        
                except Exception as e:
                    print(f"ERROR: Failed to generate mathematical equations: {str(e)}")
                    
                    if 'math_generation_span' in locals():
                        math_generation_span.update(
                            output={"error": str(e), "generation_success": False}
                        )
                        math_generation_span.end()
                    
                    # Add fallback text content
                    content_sections.append({
                        "type": "text",
                        "content": f"Mathematical equations related to {topic} would be displayed here.",
                        "editable": True,
                        "component_name": component
                    })
                    
            elif "diagram" in component.lower()  or "flow" in component.lower() or "table" in component.lower() or "illustration" in component.lower():
                # Generate mermaid diagram
                heading = f"**{component}**"
                import json
                context_block = get_top_image_contexts(topic, content_type, component, top_k=10)
                
                try:
                    mer = f"""
                        You are an expert in generating Mermaid diagrams in JSON format.

                        Topic: "{topic}"
                        Diagram Type: "{component}" or relevant type
                        Reference diagram Context:\n{context_block}
                        Reference text Context:\n {research_context}

                        Now, generate a JSON output that corresponds to a mermaid diagram representing the above topic, following the referenced structure and insights.

                        Task:
                        – Generate a meaningful Mermaid diagram that accurately represents the topic based on the provided reference context  
                        – Use less than 15  nodes, layout direction, and branches necessary
                        – Focus on clarity, compactness, and contextual relevance
                        – Ensure node labels are contextual and non-repetitive

                        Output Format:
                        Return ONLY a valid JSON object in the following format:  
                        {{ "code": "<MERMAID_CODE>" }}

                        Do NOT:
                        - Repeat earlier structures
                        - Use markdown or ``` syntax
                        - Provide explanation
                        - Overcomplicate with too many branches.
                        
                        Return only valid Mermaid code without markdown blocks.
                        """
                    
                    # Log the prompt
                    langfuse.create_prompt(
                        name="diagram_generation_prompt",
                        prompt=mer,
                        type="text"
                    )
                    
                    diagram_generation_span = langfuse.span(
                        trace_id=trace.id,
                        name="diagram_generation",
                        input={
                            "prompt": mer,
                            "context_block_length": len(str(context_block)) if context_block else 0
                        }
                    )
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=mer,
                        config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8, top_p=0.9)
                    )
                    response_text = response.text.strip()

# Remove markdown fences if present
                    if response_text.startswith("```") and response_text.endswith("```"):
                        response_text = "\n".join(line for line in response_text.splitlines() if not line.strip().startswith("```")).strip()

                    try:
                        mermaid_json = json.loads(response_text)
                        if not isinstance(mermaid_json, dict) or "code" not in mermaid_json:
                            raise ValueError("Invalid JSON structure or missing 'code' field")
                        
                        mermaid_code = mermaid_json["code"]

                    except json.JSONDecodeError as json_err:
                        print(f"DEBUG: JSON parsing failed: {json_err}")
                        print(f"DEBUG: Raw model output: {response_text!r}")

                        # Try extracting code using regex if JSON is malformed
                        import re
                        code_match = re.search(r'"code"\s*:\s*"([^"]+)"', response_text)
                        if code_match:
                            mermaid_code = code_match.group(1)
                        else:
                            raise ValueError("Could not extract 'code' from model output")

                    # mermaid_json = json.loads(response.text)
                    # mermaid_code = mermaid_json['code']
                    
                    # Generate diagram image
                    svg = generate_mermaid_diagram({"code": mermaid_code})
                    
                    if svg:
                        content_sections.append({
                            "type": "text",
                            "content": heading,
                            "editable": False,
                            "component_name": "heading"
                        })
                        
                        import time
                        timestamp = int(time.time())
                        img_filename = f"diagram_{timestamp}_{component.replace(' ', '_')}.png"
                        img_path = os.path.abspath(img_filename)
                        
                        print(f"DEBUG: Saving diagram to: {img_path}")
                        
                        try:
                            image_data = BytesIO(b64decode(svg))
                            img = Image.open(image_data)
                            
                            # Apply sizing logic
                            max_width, min_width, max_height = 700, 300, 900
                            img_w, img_h = img.size
                            aspect = img_h / img_w
                            scaled_w = max(min(img_w, max_width), min_width)
                            scaled_h = scaled_w * aspect
                            if scaled_h > max_height:
                                scaled_h = max_height
                                scaled_w = scaled_h / aspect
                                
                            img = img.resize((int(scaled_w), int(scaled_h)), Image.Resampling.LANCZOS)
                            
                            # Save with error handling
                            img.save(img_path, 'PNG', optimize=True)
                            
                            # Verify file was created
                            if os.path.exists(img_path):
                                print(f"DEBUG: Successfully saved diagram: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                                
                                diagram_generation_span.update(
                                    output={
                                        "mermaid_code": mermaid_code,
                                        "image_path": img_path,
                                        "image_size": os.path.getsize(img_path),
                                        "generation_success": True
                                    }
                                )
                                
                                content_sections.append({
                                    "type": "image",
                                    "content": img_path,
                                    "editable": True,
                                    "component_name": component,
                                    "mermaid_code": mermaid_code
                                })
                            else:
                                raise Exception("File was not created successfully")
                                
                        except Exception as save_error:
                            print(f"DEBUG: Error saving diagram: {save_error}")
                            raise save_error
                            
                    else:
                        raise Exception("SVG generation returned None")
                    
                    diagram_generation_span.end()
                        
                except Exception as e:
                    print(f"DEBUG: Diagram generation failed: {e}")
                    
                    if 'diagram_generation_span' in locals():
                        diagram_generation_span.update(
                            output={"error": str(e), "generation_success": False}
                        )
                        diagram_generation_span.end()
                    
                    # Fallback to text
                    fallback_prompt = f"""
                     Summarize what the **{component}** for "{topic}" should ideally contain.

                    Use this context:
                    {knowledge_base}

                    Instructions:
                    - Start with a short **heading** that reflects the {component} type and topic
                    - Then provide 3–5 bullet points
                    - Each bullet should describe one clear, useful idea
                    - Use plain English (no jargon, no markdown, no fluff)
                    - No explanation or preamble
                    """
                    
                    # Log fallback prompt
                    langfuse.create_prompt(
                        name="diagram_fallback_prompt",
                        prompt=fallback_prompt,
                        type="text"
                    )
                    
                    fallback_span = langfuse.span(
                        trace_id=trace.id,
                        name="diagram_fallback_generation",
                        input={"fallback_reason": str(e), "prompt": fallback_prompt}
                    )
                    
                    fallback_content = content_enrichment_agent.generate_reply([
                        {
                            "role": "user", 
                            "content": fallback_prompt
                        }
                    ])
                    
                    fallback_span.update(output={"fallback_content": fallback_content})
                    fallback_span.end()
                    
                    content_sections.append({
                        "type": "text",
                        "content": fallback_content,
                        "editable": True,
                        "component_name": component
                    })

            # For real-world photo generation  
            elif component.lower() == "real-world photo":
                try:
                    unique_seed = np.random.randint(1000, 9999)
                    timestamp = int(time.time())
                    
                    photo_prompt = (
                        f"Generate a realistic, high-resolution photograph that visually represents the real-world application of '{topic}'.\n"
                        f"The image should resemble an authentic, unstaged snapshot of a practical scenario where '{topic}' is being used or demonstrated in action.\n"
                        f"Avoid logos, branding, or any text overlays.\n"
                        f"Refer to the following similar real-world scenes for inspiration:\n\n{context_block}\n\n"
                        f"Ensure the final output maintains a natural look and clearly communicates the essence of '{topic}' without needing any explicit labels."
                    )
                    
                    # Log the prompt
                    langfuse.create_prompt(
                        name="real_world_photo_generation_prompt",
                        prompt=photo_prompt,
                        type="text"
                    )
                    
                    photo_generation_span = langfuse.span(
                        trace_id=trace.id,
                        name="real_world_photo_generation",
                        input={
                            "prompt": photo_prompt,
                            "unique_seed": unique_seed,
                            "timestamp": timestamp
                        }
                    )
                    
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-preview-image-generation",
                        contents=photo_prompt,
                        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
                    )

                    photo_generated = False
                    for idx, part in enumerate(response.candidates[0].content.parts):
                        if part.inline_data:
                            # Create unique filename with absolute path
                            img_filename = f"photo_{timestamp}_{unique_seed}_{idx}.png"
                            img_path = os.path.abspath(img_filename)
                            
                            print(f"DEBUG: Saving photo to: {img_path}")
                            
                            try:
                                photo_data = BytesIO(part.inline_data.data)
                                photo_img = Image.open(photo_data)
                                
                                # Convert to RGB if needed
                                if photo_img.mode != 'RGB':
                                    photo_img = photo_img.convert('RGB')
                                
                                # Save with error handling
                                photo_img.save(img_path, 'PNG', optimize=True)
                                
                                # Verify file was created
                                if os.path.exists(img_path):
                                    print(f"DEBUG: Successfully saved photo: {img_path} (size: {os.path.getsize(img_path)} bytes)")
                                    
                                    photo_generation_span.update(
                                        output={
                                            "image_path": img_path,
                                            "image_size": os.path.getsize(img_path),
                                            "generation_success": True,
                                            "image_index": idx
                                        }
                                    )
                                    
                                    content_sections.append({
                                        "type": "image",
                                        "content": img_path,
                                        "editable": True,
                                        "component_name": component
                                    })
                                    photo_generated = True
                                    break
                                else:
                                    raise Exception("Photo file was not created successfully")
                                    
                            except Exception as save_error:
                                print(f"DEBUG: Error saving photo: {save_error}")
                                continue
                    
                    if not photo_generated:
                        raise Exception("No photo was generated from the response")
                    
                    photo_generation_span.end()
                
                        
                except Exception as e:
                    print(f"DEBUG: Photo generation failed: {e}")
                    
                    if 'photo_generation_span' in locals():
                        photo_generation_span.update(
                            output={"error": str(e), "generation_success": False}
                        )
                        photo_generation_span.end()
                
            elif component.lower() == "graph" or "graph" in component.lower():
                
              
                try:
                    topic = topic
                    content_type = content_type.strip()
                    knowledge_base = knowledge_base if knowledge_base else ""

                    # ===== STEP 1: Get graph topic & type =====
                    graph_prompt = f"""
                    You are a senior market research analyst and data visualization strategist.

                    Your goal: For the given topic and knowledge base, create a *meaningful, insightful, and data-rich* graph idea 
                    that captures trends, comparisons, or forecasts that would be valuable to decision-makers.

                    Rules for graph_topic:
                    - It must be specific and research-oriented (e.g., "Global AI Market Size Projection 2024–2030").
                    - It should relate to measurable trends, comparisons, or statistics.
                    - Avoid generic terms like "AI statistics" or "Graph about AI".

                    Rules for graph_type:
                    - Choose the type that best represents the data (Pie Chart, Bar Chart, Line Chart, Area Chart, Scatter Plot, Histogram).

                    Topic: "{topic}"
                    Content Type: "{content_type}"
                    Knowledge Base: "{knowledge_base}"

                    Respond in JSON with:
                    {{
                        "graph_topic": "...",
                        "graph_type": "..."
                    }}
                    No extra data or text or space
                    """
                    graph_type_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=graph_prompt,
                        config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8, top_p=0.9)
                    )

                    import json
                    raw_text = graph_type_response.text.strip()

# Remove markdown fences if model added them
                    if raw_text.startswith("```") and raw_text.endswith("```"):
                        raw_text = "\n".join(
                            line for line in raw_text.splitlines()
                            if not line.strip().startswith("```")
                        ).strip()

                    try:
                        graph_info = json.loads(raw_text)
                        if not isinstance(graph_info, dict):
                            raise ValueError("Invalid JSON structure: not a dictionary")
                        if "graph_topic" not in graph_info or "graph_type" not in graph_info:
                            raise ValueError("Missing required keys in JSON")

                        graph_topic = graph_info["graph_topic"]
                        graph_type = graph_info["graph_type"]

                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Failed to parse JSON: {e}")
                        print(f"DEBUG: Raw model output: {raw_text!r}")

                        # Try regex-based extraction
                        import re
                        topic_match = re.search(r'"graph_topic"\s*:\s*"([^"]+)"', raw_text)
                        type_match = re.search(r'"graph_type"\s*:\s*"([^"]+)"', raw_text)

                        if topic_match and type_match:
                            graph_topic = topic_match.group(1)
                            graph_type = type_match.group(1)
                        else:
                            raise ValueError("Could not extract graph_topic and graph_type from model output")


                    # ===== STEP 2: Serper API Google Image Search =====
                    import requests, os, time , json

                    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
                    if not SERPER_API_KEY:
                        raise ValueError("SERPER_API_KEY not set in environment variables.")

                    query = f"{graph_topic} {graph_type} chart"

                    url = "https://google.serper.dev/images"
                    headers = {
                        "X-API-KEY": SERPER_API_KEY,
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "q": query,
                        "num": 5
                    }

                    resp = requests.post(url, headers=headers, json=payload)
                    data = resp.json()

                    from PIL import Image, UnidentifiedImageError
                    import io

                    if "images" in data and data["images"]:
                        image_url = data["images"][0]["imageUrl"]  # pick the top result
                        img_filename = f"graph_{int(time.time())}.png"
                        img_path = os.path.abspath(img_filename)

                        try:
                            # Download
                            img_data = requests.get(image_url, timeout=10).content

                            # Verify before saving
                            img_buffer = io.BytesIO(img_data)
                            try:
                                with Image.open(img_buffer) as test_img:
                                    test_img.verify()  # quick integrity check
                            except (UnidentifiedImageError, OSError) as img_err:
                                print(f"DEBUG: Skipping invalid image from {image_url}: {img_err}")
                                continue  # skip to next image/component

                            # If valid, reopen for saving
                            img_buffer.seek(0)
                            with Image.open(img_buffer) as img:
                                img.save(img_path, "PNG", optimize=True)

                            if os.path.exists(img_path):
                                content_sections.append({
                                    "type": "image",
                                    "content": img_path,
                                    "editable": True,
                                    "component_name": component,
                                    "graph_topic": graph_topic,
                                    "graph_type": graph_type
                                })
                                print(f"DEBUG: Successfully saved and added valid image: {img_path}")
                            else:
                                print(f"DEBUG: Image file not created: {img_path}")
                        except Exception as e:
                            print(f"DEBUG: Graph generation failed: {e}")

                except Exception as e:
                            print(f"DEBUG: Graph generation failed: {e}")


                        
                        # End component span
        component_span.update(
                output={"content_sections_added": len([cs for cs in content_sections if cs.get("component_name") == component])}
                        )
        component_span.end()
        
        # Create slide object
        slide = Slide(slide_title.strip(), content_sections, slide_number)
        slides.append(slide)
        
        # End slide span
        slide_span.update(
            output={
                "slide_title": slide_title.strip(),
                "content_sections_count": len(content_sections),
                "slide_created": True
            }
        )
        slide_span.end()
        
        slide_number += 1
    
    # End main trace
    # trace.update(
    #     output={
    #         "total_slides_generated": len(slides),
    #         "slide_numbers": list(range(1, slide_number)),
    #         "generation_success": True
    #     }
    # )
    # trace.end()
    
    return slides
def render_slide(slide, slide_index):
    """Render a single slide with interactive elements"""
    
    # Calculate progress
    progress = ((slide_index + 1) / len(st.session_state.slides)) * 100
    
    # Slide container
    st.markdown(f"""
    <div class="slide-container fade-in">
        <div class="slide-header">
            {slide.title}
            <div class="slide-number">{slide.slide_number}/{len(st.session_state.slides)}</div>
        </div>
        <div class="progress-bar" style="width: {progress}%"></div>
    """, unsafe_allow_html=True)
    
    # Slide content
    st.markdown('<div class="slide-content">', unsafe_allow_html=True)
    
    # Render each content section
    for section_index, section in enumerate(slide.content_sections):
        section_id = f"slide_{slide_index}_section_{section_index}"
        
        # Create editable section
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown(f'<div class="editable-section" id="{section_id}">', unsafe_allow_html=True)
            
            if section["type"] == "text":
                st.markdown(section["content"])
            elif section["type"] == "code":
                st.code(section["content"], language="python")
            elif section["type"] == "image":
                # Enhanced image display with multiple fallback methods
                try:
                    image_path = section["content"]
                    component_name = section.get('component_name', 'Generated Image')
                    
                    # Debug information
                    print(f"DEBUG: Attempting to display image: {image_path}")
                    print(f"DEBUG: File exists: {os.path.exists(image_path)}")
                    
                    if os.path.exists(image_path):
                        # Method 1: Try with PIL Image object (most reliable)
                        try:
                            with Image.open(image_path) as img:
                                # Convert to RGB if needed (handles various formats)
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                st.image(img, 
                                        caption=component_name, 
                                        use_container_width=True)
                                print(f"DEBUG: Successfully displayed image using PIL")
                                
                        except Exception as pil_error:
                            print(f"DEBUG: PIL method failed: {pil_error}")
                            
                            # Method 2: Try direct file path
                            try:
                                st.image(image_path, 
                                        caption=component_name, 
                                        use_container_width=True)
                                print(f"DEBUG: Successfully displayed image using direct path")
                                
                            except Exception as direct_error:
                                print(f"DEBUG: Direct path method failed: {direct_error}")
                                
                                # Method 3: Try reading as bytes
                                try:
                                    with open(image_path, 'rb') as img_file:
                                        img_bytes = img_file.read()
                                    
                                    st.image(img_bytes, 
                                            caption=component_name, 
                                            use_container_width=True)
                                    print(f"DEBUG: Successfully displayed image using bytes")
                                    
                                except Exception as bytes_error:
                                    print(f"DEBUG: Bytes method failed: {bytes_error}")
                                    
                                    # Method 4: Show error with file info
                                    st.error(f"Failed to display image: {image_path}")
                                    st.text(f"File size: {os.path.getsize(image_path) if os.path.exists(image_path) else 'File not found'}")
                                    st.text(f"Error details: {str(bytes_error)}")
                    else:
                        # File doesn't exist - show detailed error
                        st.error(f"Image file not found: {image_path}")
                        st.text(f"Current working directory: {os.getcwd()}")
                        st.text(f"Files in current directory: {os.listdir('.')}")
                        
                        # Try to find similar files
                        directory = os.path.dirname(image_path) or '.'
                        filename = os.path.basename(image_path)
                        if os.path.exists(directory):
                            similar_files = [f for f in os.listdir(directory) 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                            if similar_files:
                                st.text(f"Available image files in {directory}: {similar_files}")
                            
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")
                    st.text(f"Image path: {section['content']}")
                    st.text(f"Component: {section.get('component_name', 'Unknown')}")
                    
                    # Show a placeholder instead
                    st.markdown(f"""
                    <div style="
                        border: 2px dashed #ccc; 
                        padding: 40px; 
                        text-align: center; 
                        border-radius: 10px; 
                        background: #f9f9f9;
                        color: #666;
                    ">
                        📷 Image could not be displayed<br>
                        <small>{component_name}</small>
                    </div>
                    """, unsafe_allow_html=True)
            elif section["type"] == "diagram" or "flow" in section["type"]  or "chart" in section["type"] or "table" in section["type"] or "illustration" in section["type"]:
                try:
                    # Render mermaid diagram
                    st.markdown("```mermaid\n" + section["content"] + "\n```")
                except:
                    st.text(section["content"])
            else:
                st.markdown(section["content"])
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Edit button
            if st.button("✏️", key=f"edit_{section_id}", help="Edit this section"):
                st.session_state.editing_section = (slide_index, section_index)
            
            # Comment button
            if st.button("💬", key=f"comment_{section_id}", help="Add comment"):
                st.session_state.commenting_section = (slide_index, section_index)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_editing_panel():
    """Render the editing panel for selected sections"""
    if st.session_state.editing_section:
        slide_index, section_index = st.session_state.editing_section
        slide = st.session_state.slides[slide_index]
        section = slide.content_sections[section_index]
        
        st.sidebar.markdown("### ✏️ Edit Section")
        st.sidebar.markdown(f"**Component:** {section.get('component_name', 'Unknown')}")
        
        # Edit options
        edit_type = st.sidebar.selectbox("Edit Type:", [
            "Regenerate with AI",
            # "Add more detail",
            # "Simplify content",
            # "Custom instruction"
        ])
        
        if edit_type == "Custom instruction":
            custom_instruction = st.sidebar.text_area("Custom instruction:")
        else:
            custom_instruction = ""
        
        user_feedback = st.sidebar.text_area("Additional feedback or requirements:")
        
        
        if st.sidebar.button("🔄 Regenerate", type="primary"):
                with st.spinner("Regenerating content..."):
                    try:
                        # Prepare regeneration prompt
                        research_context = f"""
                        Knowledge Base: {st.session_state.knowledge_base }
                        Edit Type: {edit_type}
                        Custom Instructions: {custom_instruction}
                        User Feedback: {user_feedback}
                        Current Content: {section['content']}
                        Component Type: {section.get('component_name', 'text')}
                        """
                        
                        component_name = section.get('component_name', 'text').lower()
                        
                        # Handle different content types
                        if section['type'] == 'diagram' or 'diagram' in component_name or 'chart' in component_name or 'flow' in component_name or 'illustration' in component_name or 'graph' in component_name:
                            # Regenerate diagram/chart content
                            try:
                                context_block = get_top_image_contexts(
                                    st.session_state.get('current_topic', 'general topic'), 
                                    st.session_state.get('current_content_type', 'general'), 
                                    component_name, 
                                    top_k=10
                                )
                                
                                response = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=f"""
                                    Generate an improved Mermaid diagram based on user requirements.
                                    
                                    {research_context}
                                    
                                    Reference Context:
                                    {context_block}
                                    
                                    Create an improved diagram that:
                                    - Addresses the edit type: {edit_type}
                                    - Incorporates user feedback: {user_feedback}
                                    - Uses research findings for accuracy
                                    - Uses less than 15 nodes with clear, vertical structure
                                    - Maintains professional quality
                                    
                                    Output Format:
                                    Return ONLY a valid JSON object: {{"code": "<MERMAID_CODE>"}}
                                    
                                    Do NOT use markdown blocks or explanations.
                                    """,
                                    config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8)
                                )
                                
                                # Parse JSON response
                                response_text = response.text.strip()

# Remove markdown fences if present
                                if response_text.startswith("```") and response_text.endswith("```"):
                                    response_text = "\n".join(line for line in response_text.splitlines() if not line.strip().startswith("```")).strip()

                                try:
                                    mermaid_json = json.loads(response_text)
                                    if not isinstance(mermaid_json, dict) or "code" not in mermaid_json:
                                        raise ValueError("Invalid JSON structure or missing 'code' field")
                                    
                                    mermaid_code = mermaid_json["code"]

                                except json.JSONDecodeError as json_err:
                                    print(f"DEBUG: JSON parsing failed: {json_err}")
                                    print(f"DEBUG: Raw model output: {response_text!r}")

                                    # Try extracting code using regex if JSON is malformed
                                    import re
                                    code_match = re.search(r'"code"\s*:\s*"([^"]+)"', response_text)
                                    if code_match:
                                        mermaid_code = code_match.group(1)
                                    else:
                                        raise ValueError("Could not extract 'code' from model output")
                                
                                # Generate new diagram image
                                svg = generate_mermaid_diagram({"code": mermaid_code})
                                if svg:
                                    import time
                                    timestamp = int(time.time())
                                    img_filename = f"diagram_regenerated_{timestamp}_{component_name.replace(' ', '_')}.png"
                                    img_path = os.path.abspath(img_filename)
                                    
                                    # Convert SVG to PNG
                                    image_data = BytesIO(b64decode(svg))
                                    img = Image.open(image_data)
                                    
                                    # Resize if needed
                                    max_width, min_width, max_height = 700, 300, 900
                                    img_w, img_h = img.size
                                    aspect = img_h / img_w
                                    scaled_w = max(min(img_w, max_width), min_width)
                                    scaled_h = scaled_w * aspect
                                    if scaled_h > max_height:
                                        scaled_h = max_height
                                        scaled_w = scaled_h / aspect
                                        
                                    img = img.resize((int(scaled_w), int(scaled_h)), Image.Resampling.LANCZOS)
                                    img.save(img_path, 'PNG', optimize=True)
                                    st.image(img, caption=f"Regenerated Diagram: {component_name}", use_column_width=False)
                                    # Update section with new image path
                                    st.session_state.slides[slide_index].content_sections[section_index]['image'] = img
                                    st.session_state.slides[slide_index].content_sections[section_index]['mermaid_code'] = mermaid_code
                                    
                                else:
                                    
                                    new_content = content_enrichment_agent.generate_reply([
                                    {
                                        "role": "user",
                                        "content": f"""
                                        Create a detailed text description of what the {component_name} should contain.
                                        
                                        {research_context}
                                        
                                        Generate content that:
                                        - Describes the visual concept in clear bullet points
                                        - Addresses the edit type: {edit_type}
                                        - Incorporates user feedback: {user_feedback}
                                        - Uses research findings for accuracy
                                        """
                                    }
                                ])
                                    st.session_state.slides[slide_index].content_sections[section_index]['content'] = new_content.strip()
                                    st.session_state.slides[slide_index].content_sections[section_index]['type'] = 'text'
                            
                                    
                            except Exception as diagram_error:
                                # st.warning(f"Diagram regeneration failed: {diagram_error}. Generating text alternative.")
                                # Fallback to text description
                                new_content = content_enrichment_agent.generate_reply([
                                    {
                                        "role": "user",
                                        "content": f"""
                                        Create a detailed text description of what the {component_name} should contain.
                                        
                                        {research_context}
                                        
                                        Generate content that:
                                        - Describes the visual concept in clear bullet points
                                        - Addresses the edit type: {edit_type}
                                        - Incorporates user feedback: {user_feedback}
                                        - Uses research findings for accuracy
                                        """
                                    }
                                ])
                                st.session_state.slides[slide_index].content_sections[section_index]['content'] = new_content.strip()
                                st.session_state.slides[slide_index].content_sections[section_index]['type'] = 'text'
                        
                        elif section['type'] == 'image' and 'mathematical' in component_name:
                            # Regenerate mathematical equation
                            try:
                                # Generate new equation
                                equation_response = content_generator.generate_reply([
                                    {
                                        "role": "user",
                                        "content": f"""
                                        Generate an improved mathematical equation based on user requirements.
                                        
                                        {research_context}
                                        
                                        Requirements:
                                        - Create ONE clear LaTeX equation relevant to the topic
                                        - Address the edit type: {edit_type}
                                        - Incorporate user feedback: {user_feedback}
                                        - Use proper LaTeX syntax without \\text commands
                                        - Do not wrap in $$ symbols
                                        
                                        Format:
                                        EQUATION: [LaTeX equation]
                                        DESCRIPTION: [brief description]
                                        """
                                    }
                                ])
                                
                                # Parse equation and description
                                latex_equation = ""
                                description = ""
                                
                                lines = equation_response.strip().split('\n')
                                for line in lines:
                                    if line.startswith("EQUATION:"):
                                        latex_equation = line.replace("EQUATION:", "").strip()
                                    elif line.startswith("DESCRIPTION:"):
                                        description = line.replace("DESCRIPTION:", "").strip()
                                
                                if not latex_equation:
                                    latex_equation = "F(x) = \\sum_{i=1}^n a_i x^i"  # Fallback
                                
                                # Generate new image
                                import time
                                timestamp = int(time.time())
                                img_filename = f"equation_regenerated_{timestamp}.png"
                                img_path = os.path.abspath(img_filename)
                                
                                rendered_path = render_latex_to_image(latex_equation, img_path)
                                img.save(rendered_path, 'PNG', optimize=True)
                                st.image(img, caption=f"Regenerated Diagram: {component_name}", use_column_width=False)
                                
                                if rendered_path and os.path.exists(rendered_path):
                                    st.session_state.slides[slide_index].content_sections[section_index]['image'] = img
                                    
                                    # Update description if it exists in the next section
                                    if section_index + 1 < len(st.session_state.slides[slide_index].content_sections):
                                        next_section = st.session_state.slides[slide_index].content_sections[section_index + 1]
                                        if 'description' in next_section.get('component_name', ''):
                                            next_section['content'] = description
                                else:
                                    raise Exception("Equation image generation failed")
                                    
                            except Exception as math_error:
                                st.warning(f"Mathematical equation regeneration failed: {math_error}. Using text fallback.")
                                fallback_content = f"**Mathematical Concepts** (Regenerated)\n\n{user_feedback}\n\nMathematical representation and analysis related to the topic."
                                st.session_state.slides[slide_index].content_sections[section_index]['content'] = fallback_content
                                st.session_state.slides[slide_index].content_sections[section_index]['type'] = 'text'
                        
                        elif section['type'] == 'image' and 'photo' in component_name:
                            # Regenerate real-world photo
                            try:
                                context_block = get_top_image_contexts(
                                    st.session_state.get('current_topic', 'general topic'), 
                                    st.session_state.get('current_content_type', 'general'), 
                                    component_name, 
                                    top_k=10
                                )
                                
                                photo_prompt = f"""
                                Generate an improved realistic photograph based on user requirements.
                                
                                Topic: {st.session_state.get('current_topic', 'general topic')}
                                Edit Type: {edit_type}
                                User Feedback: {user_feedback}
                                
                                Create a high-resolution photograph that:
                                - Shows real-world application of the topic
                                - Addresses the specific improvements requested
                                - Maintains professional quality
                                - Avoids logos, branding, or text overlays
                                
                                Reference context for inspiration:
                                {context_block}
                                """
                                
                                response = client.models.generate_content(
                                    model="gemini-2.0-flash-preview-image-generation",
                                    contents=photo_prompt,
                                    config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
                                )
                                
                                photo_generated = False
                                for idx, part in enumerate(response.candidates[0].content.parts):
                                    if part.inline_data:
                                        import time
                                        timestamp = int(time.time())
                                        img_filename = f"photo_regenerated_{timestamp}_{idx}.png"
                                        img_path = os.path.abspath(img_filename)
                                        
                                        photo_data = BytesIO(part.inline_data.data)
                                        photo_img = Image.open(photo_data)
                                        
                                        if photo_img.mode != 'RGB':
                                            photo_img = photo_img.convert('RGB')
                                        
                                        photo_img.save(img_path, 'PNG', optimize=True)
                                        
                                        if os.path.exists(img_path):
                                            
                                            st.image(photo_img, caption=f"Regenerated Diagram: {component_name}", use_column_width=False)
                                            st.session_state.slides[slide_index].content_sections[section_index]['image'] = img_path
                                            photo_generated = True
                                            break
                                
                                if not photo_generated:
                                    raise Exception("Photo generation failed")
                                    
                            except Exception as photo_error:
                                st.warning(f"Photo regeneration failed: {photo_error}. Using text description.")
                                fallback_content = f"**Real-World Application** (Regenerated)\n\n{user_feedback}\n\nDescription of practical applications and real-world usage scenarios."
                                st.session_state.slides[slide_index].content_sections[section_index]['content'] = fallback_content
                                st.session_state.slides[slide_index].content_sections[section_index]['type'] = 'text'
                        
                        elif section['type'] == 'code':
                            # Regenerate code content
                            template = PromptTemplate(
                                template="""
                                Generate improved Python code based on user requirements.
                                
                                Current code: {current_code}
                                Edit type: {edit_type}
                                User feedback: {user_feedback}
                                Topic: {topic}
                                
                                Requirements:
                                - Fully runnable Python code (no syntax errors)
                                - ≤ 30 lines
                                - Address the specific improvements requested
                                - Use functions/classes if useful
                                - No markdown formatting
                                
                                {format_instructions}
                                """,
                                input_variables=["current_code", "edit_type", "user_feedback", "topic"],
                                partial_variables={"format_instructions": parser.get_format_instructions()}
                            )
                            
                            prompt = template.format(
                                current_code=section['content'],
                                edit_type=edit_type,
                                user_feedback=user_feedback,
                                topic=st.session_state.get('current_topic', 'programming')
                            )
                            
                            raw_output = content_generator.generate_reply([{
                                "role": "user",
                                "content": prompt
                            }])
                            
                            try:
                                parsed = parser.parse(raw_output)
                                new_content = parsed.code
                            except Exception:
                                # Fallback to regex extraction
                                match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
                                new_content = match.group(1).strip() if match else raw_output.strip()
                            
                            st.session_state.slides[slide_index].content_sections[section_index]['content'] = new_content
                        
                        else:
                            # Regenerate text content (default)
                            new_content = content_enrichment_agent.generate_reply([
                                {
                                    "role": "user",
                                    "content": f"""
                                    Improve this content based on user requirements.
                                    
                                    {research_context}
                                    
                                    Generate improved content that:
                                    - Addresses the edit type: {edit_type}
                                    - Incorporates user feedback: {user_feedback}
                                    - Uses research findings for accuracy
                                    - Maintains professional presentation format
                                    - Keeps appropriate length for slides
                                    - Add 4–5 **Key Points** (≤ 1.5 lines each, aim for 1 line)
                                    - Add 2–3 **Real-World Applications** (1–1.5 lines each)
                                    - Only use external knowledge
                                    - Keep tone professional and minimal ( audience)
                                    
                                    Return only the improved content without explanations.
                                    """
                                }
                            ])
                            
                            st.session_state.slides[slide_index].content_sections[section_index]['content'] = new_content.strip()
                        
                        st.success("✅ Content regenerated successfully!")
                        st.session_state.editing_section = None
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error regenerating content: {str(e)}")
                        print(f"DEBUG: Regeneration error details: {type(e).__name__}: {e}")

# Cancel button

        # Cancel button
        if st.sidebar.button("❌ Cancel"):
            st.session_state.editing_section = None
            st.rerun()

def render_comment_panel():
    """Render the comment panel for selected sections"""
    if hasattr(st.session_state, 'commenting_section') and st.session_state.commenting_section:
        slide_index, section_index = st.session_state.commenting_section
        
        st.sidebar.markdown("### 💬 Add Comment")
        
        comment_text = st.sidebar.text_area("Your comment:")
        comment_type = st.sidebar.selectbox("Comment type:", [
            "General feedback",
            "Improvement suggestion",
            "Question",
            "Clarification needed",
            "Research gap"
        ])
        
        if st.sidebar.button("📝 Add Comment", type="primary"):
            if comment_text.strip():
                # Initialize comments if not exists
                if slide_index not in st.session_state.comments:
                    st.session_state.comments[slide_index] = {}
                if section_index not in st.session_state.comments[slide_index]:
                    st.session_state.comments[slide_index][section_index] = []
                
                # Add comment
                st.session_state.comments[slide_index][section_index].append({
                    "text": comment_text.strip(),
                    "type": comment_type,
                    "timestamp": st.session_state.get('timestamp', 0)
                })
                
                st.success("Comment added!")
                st.session_state.commenting_section = None
                st.rerun()
            else:
                st.warning("Please enter a comment.")
        
        if st.sidebar.button("❌ Cancel"):
            st.session_state.commenting_section = None
            st.rerun()

def render_presentation_controls():
    """Render presentation navigation and controls"""
    if st.session_state.slides:
        # Navigation controls
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First", disabled=st.session_state.current_slide == 0):
                st.session_state.current_slide = 0
                st.rerun()
        
        with col2:
            if st.button("◀️ Previous", disabled=st.session_state.current_slide == 0):
                st.session_state.current_slide -= 1
                st.rerun()
        
        with col3:
            # Slide selector
            slide_options = [f"Slide {i+1}: {slide.title[:30]}..." if len(slide.title) > 30 else f"Slide {i+1}: {slide.title}" 
                           for i, slide in enumerate(st.session_state.slides)]
            selected_slide = st.selectbox(
                "Current slide:",
                options=range(len(st.session_state.slides)),
                format_func=lambda x: slide_options[x],
                index=st.session_state.current_slide,
                key="slide_selector"
            )
            if selected_slide != st.session_state.current_slide:
                st.session_state.current_slide = selected_slide
                st.rerun()
        
        with col4:
            if st.button("▶️ Next", disabled=st.session_state.current_slide >= len(st.session_state.slides) - 1):
                st.session_state.current_slide += 1
                st.rerun()
        
        with col5:
            if st.button("⏭️ Last", disabled=st.session_state.current_slide >= len(st.session_state.slides) - 1):
                st.session_state.current_slide = len(st.session_state.slides) - 1
                st.rerun()

def render_slide_with_comments(slide, slide_index):
    """Render slide with comments displayed"""
    render_slide(slide, slide_index)
    
    # Display comments for this slide
    if slide_index in st.session_state.comments:
        st.markdown("### 💬 Comments")
        for section_index, comments in st.session_state.comments[slide_index].items():
            if comments:
                section_name = slide.content_sections[section_index].get('component_name', f'Section {section_index + 1}')
                st.markdown(f"**{section_name}:**")
                
                for comment in comments:
                    st.markdown(f"""
                    <div class="comment-box">
                        <strong>{comment['type']}:</strong> {comment['text']}
                    </div>
                    """, unsafe_allow_html=True)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re

# ---------- helper ----------
def boldify(text: str) -> str:
    """
    Convert **text** to <b>text</b> for ReportLab Paragraphs.
    Keeps line-breaks as <br/>.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text.replace('\n', '<br/>')


from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image as PILImage
import os

import uuid

# ---------- helper ----------
def boldify(text: str) -> str:
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(text))

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re

# ---------- helper ----------
def boldify(text: str) -> str:
    """
    Convert **text** to <b>text</b> for ReportLab Paragraphs.
    Keeps line-breaks as <br/>.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text.replace('\n', '<br/>')

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
def export_to_pdf():
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # ---------- styles ----------
        title_style = ParagraphStyle(
    name='SlideTitle',
    fontSize=16,              # ↓ smaller
    leading=20,               # line-height
    spaceAfter=10,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold',
    wordWrap='CJK'            # helps long titles wrap
)
        body_style = ParagraphStyle(
            name='Body',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            spaceAfter=6
        )
        code_style = ParagraphStyle(
            name='Code',
            fontSize=9,
            fontName='Courier',
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=8,
            backColor=colors.lightgrey
        )

        # ---------- slides ----------
        for slide_idx, slide in enumerate(st.session_state.slides):
            # ---- slide title ----
            story.append(Paragraph(slide.title, title_style))
            story.append(Spacer(1, 12))

            # ---- slide content ----
            for sec in slide.content_sections:
                if sec["type"] == "text":
                    story.append(Paragraph(boldify(sec["content"]), body_style))
                elif sec["type"] == "code":
                    story.append(Preformatted(sec["content"], code_style))
                    story.append(Spacer(1, 6))
                elif sec["type"] == "image":
                    img_path = sec["content"]
                    if os.path.exists(img_path):
                        try:
                            max_w, max_h = 5*inch, 4*inch
                            img = PILImage.open(img_path)
                            w, h = img.size
                            scale = min(max_w/w, max_h/h)
                            story.append(
                                RLImage(img_path,
                                        width=w*scale,
                                        height=h*scale)
                            )
                            story.append(Spacer(1, 6))
                        except Exception:
                            story.append(Paragraph("[Image not rendered]", body_style))
            story.append(PageBreak())

        # ---------- references slide ----------
        # story.append(Paragraph("References", title_style))
        # story.append(Spacer(1, 12))

        kb = st.session_state.knowledge_base
        refs = kb
        ref_text = ""

        # collect all definitions, concepts, etc. as bullet list
        # for key, items in refs.items():
        #     for item in items:
        #         ref_text += f"• {boldify(str(item))}<br/>"

        # story.append(Paragraph(ref_text, body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border-radius: 20px; margin-bottom: 30px;">
        <h1 style="margin: 0; font-size: 3em;">🎯 AI Presentation Creator</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">Create stunning presentations with AI-powered research</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("### 🎮 Controls")
        
        # Topic input and generation
        if not st.session_state.slides:
            st.markdown("#### Generate Presentation")
            topic = st.text_input("Enter your topic:", placeholder="e.g., Machine Learning, Generative AI")
            
            depth_level = st.selectbox("Audience level:", [
                "beginner", 
                "intermediate", 
                "advanced", 
                "expert"
            ], index=1)
            
            # research_focus = st.multiselect(
            #     "Research focus:",
            #     ["Academic Papers", "Industry Blogs", "Case Studies", "Technical Documentation", "Latest Trends"],
            #     default=["Academic Papers", "Industry Blogs", "Latest Trends"]
            # )
            
            if st.button("🚀 Generate Presentation", type="primary"):
                if topic.strip():
                    with st.spinner("🔍 Researching and generating slides..."):
                        try:
                            slides = generate_presentation_slides(topic, depth_level)
                            st.session_state.slides = slides
                            st.session_state.current_slide = 0
                            st.success(f"✅ Generated {len(slides)} slides!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error generating presentation: {str(e)}")
                else:
                    st.warning("Please enter a topic.")
        
        # Presentation controls when slides exist
        if st.session_state.slides:
            st.markdown("#### Navigation")
            render_presentation_controls()
            
            st.markdown("---")
            
            # Mode toggle
            # presentation_mode = st.toggle("📺 Presentation Mode", value=st.session_state.presentation_mode)
            # if presentation_mode != st.session_state.presentation_mode:
            #     st.session_state.presentation_mode = presentation_mode
            #     st.rerun()
            
            # Export options
            st.markdown("#### Export")
            
           
            if st.button("📄 Export to PDF"):
                pdf_data = export_to_pdf()
                if pdf_data:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_data,
                        file_name=f"presentation.pdf",
                        mime="application/pdf"
                    )
            # JSON export (for backup/sharing)
            # if st.button("💾 Export Data"):
            #     export_data = {
            #         "slides": [
            #             {
            #                 "title": slide.title,
            #                 "content_sections": slide.content_sections,
            #                 "slide_number": slide.slide_number
            #             }
            #             for slide in st.session_state.slides
            #         ],
            #         "knowledge_base": st.session_state.knowledge_base,
            #         "comments": st.session_state.comments
            #     }
                
            #     st.download_button(
            #         "⬇️ Download JSON",
            #         data=json.dumps(export_data, indent=2),
            #         file_name=f"presentation_data_{topic.replace(' ', '_')}.json",
            #         mime="application/json"
            #     )
            
            # Reset option
            st.markdown("---")
            if st.button("🔄 New Presentation", type="secondary"):
                st.session_state.slides = []
                st.session_state.current_slide = 0
                st.session_state.editing_section = None
                st.session_state.comments = {}
                st.session_state.knowledge_base = {}
                st.rerun()
        
        # Render editing/comment panels
        render_editing_panel()
        render_comment_panel()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if st.session_state.slides:
        # Display current slide
        current_slide = st.session_state.slides[st.session_state.current_slide]
        
        if st.session_state.presentation_mode:
            # Full-screen presentation mode
            st.markdown("""
            <style>
            .main > div {
                padding-top: 1rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            render_slide(current_slide, st.session_state.current_slide)
            
            # Presentation navigation at bottom
            st.markdown("""
            <div class="slide-navigation">
                Use sidebar controls to navigate
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Edit mode with comments
            render_slide_with_comments(current_slide, st.session_state.current_slide)
            
            # Research context panel
            if st.session_state.knowledge_base:
                with st.expander("🔬 Research Context"):
                    kb = st.session_state.knowledge_base
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'definitions' in kb:
                            st.markdown("**Key Definitions:**")
                            for definition in kb['definitions'][:3]:
                                st.markdown(f"• {definition}")
                        
                        if 'core_concepts' in kb:
                            st.markdown("**Core Concepts:**")
                            for concept in kb['core_concepts'][:3]:
                                st.markdown(f"• {concept}")
                    
                    with col2:
                        if 'applications' in kb:
                            st.markdown("**Applications:**")
                            for application in kb['applications'][:3]:
                                st.markdown(f"• {application}")
                        
                        if 'latest_trends' in kb:
                            st.markdown("**Latest Trends:**")
                            for trend in kb['latest_trends'][:3]:
                                st.markdown(f"• {trend}")
            
            # Slide overview
            with st.expander("📋 Slide Overview"):
                for i, slide in enumerate(st.session_state.slides):
                    indicator = "➡️" if i == st.session_state.current_slide else "📄"
                    if st.button(f"{indicator} {slide.title}", key=f"overview_{i}"):
                        st.session_state.current_slide = i
                        st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
