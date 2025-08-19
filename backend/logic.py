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

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors


# --- Pydantic Models ---
class ContentSection(BaseModel):
    type: str
    content: str
    editable: bool
    component_name: str
    mermaid_code: str | None = None
    graph_topic: str | None = None
    graph_type: str | None = None

class Slide(BaseModel):
    title: str
    content_sections: List[ContentSection]
    slide_number: int

class Presentation(BaseModel):
    slides: List[Slide]
    knowledge_base: dict


class CodeSnippet(BaseModel):
    code: str = Field(..., description="Concise Python code snippet")

parser = PydanticOutputParser(pydantic_object=CodeSnippet)

# --- Environment and API Keys ---
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# --- Configuration Loading ---
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
    print(f"Configuration file not found: {e}")
    raise

# --- AI Agents ---
# (Agent definitions are omitted for brevity, they are the same as in final.py)
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


# --- Helper Functions ---

def generate_mermaid_diagram(payload: dict, vm_ip: str = "40.81.228.142:5500") -> str:
    url = f"http://{vm_ip}/render-mermaid/"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def get_top_image_contexts(topic, content_type, component, top_k=10):
    # This function needs to be adapted to find embedding files
    # For now, we assume they are in a relative path
    base_path = "embeddings_dino"
    if component.lower() == "Real-World Photo":
        emb_file = os.path.join(base_path, "real_images.npy")
        path_file = os.path.join(base_path, "real_images_paths.txt")
        query = f"Extract the relevant data for generating {component} on {content_type} of {topic}"
    else:
        emb_file = os.path.join(base_path, "diagrams.npy")
        path_file = os.path.join(base_path, "diagrams_paths.txt")
        query = f"Extract the relevant data on {content_type} of {topic} for generating {component}"

    if not os.path.exists(emb_file) or not os.path.exists(path_file):
        print(f"Warning: Embedding file not found: {emb_file} or {path_file}. Skipping context retrieval.")
        return ""

    image_emb = np.load(emb_file)
    with open(path_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]

    if len(image_paths) != len(image_emb):
        raise ValueError("Mismatch between image paths and embedding vectors.")

    text_encoder = SentenceTransformer("all-mpnet-base-v2")
    query_emb = text_encoder.encode(query, normalize_embeddings=True)
    query_emb = torch.tensor(query_emb, dtype=torch.float32)
    image_emb = torch.tensor(image_emb, dtype=torch.float32)

    scores = torch.nn.functional.cosine_similarity(query_emb, image_emb)
    top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices.numpy()
    top_contexts = [image_paths[i] for i in top_indices]
    return "\n".join(top_contexts)


def clean_latex_equation(math_expr: str) -> str:
    expr = math_expr.strip()
    expr = re.sub(r'^\$+|\$+$', '', expr)
    expr = re.sub(r'^\\begin{equation}|\\end{equation}$', '', expr)
    expr = re.sub(r'^\\begin{align}|\\end{align}$', '', expr)
    expr = re.sub(r'\\text{([^}]*)}', r'\\mathrm{\1}', expr)
    expr = re.sub(r'\\textrm{([^}]*)}', r'\\mathrm{\1}', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr

def render_latex_to_image(math_expr: str, output_path="equation.png") -> str:
    # Note: output_path should be relative to the static directory
    try:
        expr = clean_latex_equation(math_expr)
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
        ax.axis('off')
        ax.text(0.5, 0.5, f"${expr}$", horizontalalignment='center', verticalalignment='center', fontsize=24, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        plt.tight_layout(pad=0.2)

        # Save to static directory
        full_path = os.path.join("static", output_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        fig.savefig(full_path, bbox_inches='tight', transparent=False, dpi=300, facecolor='white', edgecolor='none')
        plt.close(fig)
        # Return the path relative to the static dir, so the frontend can use it
        return output_path
    except Exception as e:
        print(f"ERROR: Failed to render LaTeX equation: {str(e)}")
        return ""


def generate_presentation_slides(topic: str, depth_level: str = "intermediate") -> dict:
    """Generate slides and return as a dictionary."""

    serper_search(topic, 20)

    prompt = f"Select the appropriate content-types from {list(content_types_dict.keys())} for the topic \"{topic}\""
    output1 = content_type_Selector.generate_reply(messages=[{"role": "user", "content": prompt}])
    selected_content_types = [c.strip() for c in output1.split(",")]

    v = {k: content_types_dict[k] for k in selected_content_types if k in content_types_dict}
    prompt2 = f'''For the content-types {output1}, the default selected components are: {v}. Based on the full component list {available_components}, suggest edits.'''
    component_dict_str = component_type_Selector.generate_reply(messages=[{"role": "user", "content": prompt2}])

    try:
        component_dict = eval(component_dict_str)
    except:
        component_dict = v

    search_and_embed(f"{topic}", max_images=100)

    slides_data = []
    knowledge_base_agg = {}
    slide_number = 1

    for content_type, components in component_dict.items():
        query = f"generate the context for {content_type} of {topic}"
        knowledge_base = query_vector_db(query)
        knowledge_base_agg[content_type] = knowledge_base

        prompt1 = f"Generate a professional slide title for topic '{topic}' focused on '{content_type}'. Use this research context: {knowledge_base}. Return only the title."
        slide_title = content_enrichment_agent.generate_reply([{"role": "user", "content": prompt1}])

        content_sections_data = []
        for component in components:
            research_context = f"Knowledge Base: {knowledge_base}, Topic: {topic}, Content Type: {content_type}, Component: {component}, Depth Level: {depth_level}"

            if component.lower() == "text":
                p = f"Use this research context only as a reference: {research_context}. Generate brief, original slide content for {content_type}."
                text_content = content_generator.generate_reply([{"role": "user", "content": p}])
                content_sections_data.append({
                    "type": "text", "content": text_content.strip(), "editable": True, "component_name": component
                })
            elif component.lower() == "code snippet":
                template = PromptTemplate(
                    template="Generate a concise Python code snippet for the topic: '{topic}'. {format_instructions}",
                    input_variables=["topic"],
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )
                prompt = template.format(topic=topic)
                raw_output = content_generator.generate_reply([{"role": "user", "content": prompt}])
                try:
                    code_clean = parser.parse(raw_output).code
                except:
                    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
                    code_clean = match.group(1).strip() if match else raw_output.strip()
                content_sections_data.append({
                    "type": "code", "content": code_clean, "editable": True, "component_name": component
                })
            elif component.lower() == "mathematical equations":
                pro = f"Generate a professional and relevant mathematical equation or formula related to the topic: '{topic}'. Provide ONE clear, well-formatted LaTeX equation and a brief 1-2 sentence description. Format your response as: EQUATION: [your LaTeX equation here] DESCRIPTION: [brief description]"
                out = content_generator.generate_reply([{"role": "user", "content": pro}])
                lines = out.strip().split('\n')
                latex_equation = ""
                description = ""
                for line in lines:
                    if line.startswith("EQUATION:"):
                        latex_equation = line.replace("EQUATION:", "").strip()
                    elif line.startswith("DESCRIPTION:"):
                        description = line.replace("DESCRIPTION:", "").strip()
                if not latex_equation:
                    latex_equation = out.strip()

                img_filename = f"equation_{uuid.uuid4()}.png"
                img_path = render_latex_to_image(latex_equation, output_path=img_filename)
                if img_path:
                    content_sections_data.append({
                        "type": "image", "content": img_path, "editable": True, "component_name": component
                    })
                    if description:
                        content_sections_data.append({
                            "type": "text", "content": description, "editable": True, "component_name": f"{component}_description"
                        })

            elif "diagram" in component.lower()  or "flow" in component.lower() or "table" in component.lower() or "illustration" in component.lower():
                context_block = get_top_image_contexts(topic, content_type, component)
                mer = f"""You are an expert in generating Mermaid diagrams in JSON format. Topic: "{topic}", Diagram Type: "{component}". Reference context: {context_block}. Generate a JSON output with a "code" key for the mermaid diagram."""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=mer,
                    config=types.GenerateContentConfig(response_modalities=["TEXT"], temperature=0.8, top_p=0.9)
                )
                response_text = response.text.strip()
                mermaid_code = ""
                try:
                    mermaid_json = json.loads(response_text)
                    mermaid_code = mermaid_json['code']
                except (json.JSONDecodeError, KeyError):
                     match = re.search(r'"code"\s*:\s*"([^"]+)"', response_text)
                     if match:
                         mermaid_code = match.group(1)
                     else:
                         mermaid_code = response_text

                svg = generate_mermaid_diagram({"code": mermaid_code})
                if svg:
                    img_filename = f"diagram_{uuid.uuid4()}.png"
                    img_path = os.path.join("static", img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    try:
                        image_data = BytesIO(b64decode(svg))
                        img = Image.open(image_data)
                        img.save(img_path, 'PNG')
                        content_sections_data.append({
                            "type": "image", "content": img_filename, "editable": True, "component_name": component, "mermaid_code": mermaid_code
                        })
                    except Exception as e:
                        print(f"Failed to save diagram: {e}")
            # ... (other component types can be added here)

        slides_data.append({
            "title": slide_title.strip(),
            "content_sections": content_sections_data,
            "slide_number": slide_number
        })
        slide_number += 1

    return {"slides": slides_data, "knowledge_base": knowledge_base_agg}

def boldify(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return text.replace('\n', '<br/>')

def export_to_pdf(presentation_data: dict):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            name='SlideTitle',
            fontSize=16,
            leading=20,
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            wordWrap='CJK'
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

        for slide in presentation_data["slides"]:
            story.append(Paragraph(slide["title"], title_style))
            story.append(Spacer(1, 12))

            for sec in slide["content_sections"]:
                if sec["type"] == "text":
                    story.append(Paragraph(boldify(sec["content"]), body_style))
                elif sec["type"] == "code":
                    story.append(Preformatted(sec["content"], code_style))
                elif sec["type"] == "image":
                    # Image path is relative to static dir
                    img_path = os.path.join("static", sec["content"])
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

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None
