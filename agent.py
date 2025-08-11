from langchain_community.utilities import GoogleSerperAPIWrapper
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
import os
import requests
from urllib.parse import urlparse
from io import BytesIO
import shutil
import json

# Set your Serper API Key
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "4dd51fc28ee7339f4993df71c7e3247cc8faaf6b")
search_tool = GoogleSerperAPIWrapper(type="images")

def clean_directory(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def fetch_image_urls_from_serper(query, max_images=20):
    """Fetch image-containing pages using Langchain SerperWrapper and extract image links heuristically."""
    result_str = search_tool.run(query)
    result = search_tool.results(query)  # full structured results

    urls = []
    for r in result.get("images", []):  # THIS is the correct key
        if "imageUrl" in r:
            urls.append(r["imageUrl"])
        if len(urls) >= max_images:
            break

    return urls[:max_images]


def search_and_embed(topic, max_images=20, image_base="images", output_dir="embeddings_dino"):
    clean_directory(image_base)
    clean_directory(output_dir)

    sub_queries = {
        "diagrams": f"{topic} diagram OR chart OR illustration",
        "infographics": f"{topic} infographic",
        "real_images": f"{topic} real world photo"
    }

    total_saved = 0
    for category, sub_query in sub_queries.items():
        print(f"\n🔍 Searching for {category.replace('_', ' ')}: {sub_query}")
        category_folder = os.path.join(image_base, category)
        os.makedirs(category_folder, exist_ok=True)

        image_urls = fetch_image_urls_from_serper(sub_query, max_images)
        saved = 0

        for idx, image_url in enumerate(image_urls, 1):
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                ext = os.path.splitext(urlparse(image_url).path)[1]
                if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    ext = '.jpg'

                filename = os.path.join(
                    category_folder, f"{topic.replace(' ', '_')}_{category}_{idx}{ext}"
                )
                image.save(filename)
                print(f"✅ Saved: {filename}")
                saved += 1
                total_saved += 1
            except Exception as e:
                print(f"❌ Failed to download image {idx}: {str(e)}")
        print(f"📂 {saved} {category} images saved.\n")

    print(f"\n🎉 Done. Total images saved: {total_saved}")

    # ============ Step 2: DINOv2 Embedding ============

    BASE = Path(image_base)
    OUT = Path(output_dir); OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    for class_dir in BASE.iterdir():
        if not class_dir.is_dir(): continue
        img_paths, vecs = [], []

        for img_path in sorted(class_dir.glob("*.*")):
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)
                feats = features / features.norm(p=2, dim=-1, keepdim=True)
                vecs.append(feats.cpu().numpy()[0])
                img_paths.append(str(img_path))
            except Exception as e:
                print("ERR embedding", img_path, e)

        if vecs:
            arr = np.stack(vecs, axis=0).astype("float32")
            np.save(OUT / f"{class_dir.name}.npy", arr)
            with open(OUT / f"{class_dir.name}_paths.txt", "w") as fw:
                fw.write("\n".join(img_paths))
            print(f"✅ Saved {arr.shape} DINOv2 embeddings for {class_dir.name}")
