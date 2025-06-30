import json
import os
import re
import logging
from pathlib import Path
from datetime import datetime # Import for get_current_time function
from urllib.parse import urlparse # Ensure urlparse is imported at the top

import torch
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

# --- Global variables for model and embeddings ---
# These will be loaded once when the application starts
model = None
corpus_embeddings = None
corpus_texts = []
corpus_urls = []

# --- Global variables for Intent Data ---
intents_data = {}
training_data = [] # List of {'text': '...', 'intent': '...'}
# Using a dictionary for faster lookup by normalized text
training_phrases_map = {} # {'normalized_text': 'intent_name'}

# Define file paths relative to the current script
BASE_DIR = Path(__file__).parent
# CRITICAL FIX: Adjusted paths to include 'modules' as per your file structure
STATIC_DIR = BASE_DIR / 'modules' / 'static' 
DATA_DIR = STATIC_DIR / 'data' # Data is still inside static

EMBEDDINGS_FILE = DATA_DIR / 'scraped_content_embeddings.json'
CONTENT_FILE = DATA_DIR / 'scraped_content.json'
INTENTS_FILE = BASE_DIR / 'intents.json' # Assuming intents.json is in the same directory as app.py
TRAINING_FILE = BASE_DIR / 'training.json' # Assuming training.json is in the same directory as app.py
CHATBOT_CONFIG_FILE = BASE_DIR / 'config.json' # Assuming config.json is in the base directory

# Ensure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Flask App Setup ---
# CRITICAL FIX: Adjusted template_folder path in Flask constructor
app = Flask(__name__,
            static_folder=str(STATIC_DIR),
            template_folder=str(BASE_DIR / 'modules' / 'templates')) 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)

# --- Global variables for scraped data and homepage URL ---
scraped_data = {}
homepage_url = "" # To store the normalized homepage URL from scraped data

# --- Helper Functions ---

def load_json_data(filepath):
    """Loads JSON data from a specified file path."""
    if not filepath.exists():
        app_logger.error(f"File not found: {filepath}. Please ensure your file is in the correct directory and named correctly.")
        # Return appropriate empty structure based on file type for robustness
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else: # For training.json which is a list
            return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        app_logger.error(f"Error decoding JSON from {filepath}: {e}. Check if the JSON is valid.")
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else:
            return []
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading {filepath}: {e}")
        if 'intents.json' in str(filepath) or 'scraped_content.json' in str(filepath) or 'config.json' in str(filepath):
            return {} 
        else:
            return []

def load_scraped_content():
    """Loads the scraped content from the JSON file and identifies the homepage URL."""
    global scraped_data, homepage_url
    try:
        scraped_data = load_json_data(CONTENT_FILE)
        
        if not scraped_data:
            app_logger.error("Scraped content data is empty or could not be loaded.")
            return False
        
        app_logger.info(f"Successfully loaded scraped content from {CONTENT_FILE}. Total pages: {len(scraped_data)}")

        # Load root_url from config.json (if available, otherwise fallback)
        root_url_from_config = 'https://ktgsoftware.com/' # Default fallback
        try:
            chatbot_config = load_json_data(CHATBOT_CONFIG_FILE)
            if chatbot_config is not None: # Ensure config was loaded successfully
                root_url_from_config = chatbot_config.get('root_url', root_url_from_config)
        except Exception as e:
            app_logger.warning(f"Could not load chatbot config file from {CHATBOT_CONFIG_FILE}: {e}. Using default root URL.")

        normalized_root = normalize_url(root_url_from_config)
        
        if normalized_root in scraped_data:
            homepage_url = normalized_root
            app_logger.info(f"Identified homepage URL: {homepage_url}")
        else:
            # Fallback: try to find a likely homepage (e.g., shortest path)
            if scraped_data:
                candidate_urls = [url for url in scraped_data.keys() if urlparse(url).path in ['', '/', '/index.html']]
                if candidate_urls:
                    if normalized_root in candidate_urls:
                        homepage_url = normalized_root
                    else:
                        homepage_url = sorted(candidate_urls, key=lambda url: len(url))[0]
                elif scraped_data:
                     homepage_url = list(scraped_data.keys())[0] # Just take the first one if no clear homepage

                app_logger.warning(f"Could not find exact root URL '{normalized_root}' in scraped data. Falling back to identified homepage: {homepage_url}")
            else:
                app_logger.error("Scraped data is empty, cannot identify homepage.")

        return True
    except Exception as e:
        app_logger.error(f"An unexpected error occurred while loading scraped content: {e}")
        return False


def load_intent_data():
    """Loads intents and training phrases into global variables."""
    global intents_data, training_data, training_phrases_map

    app_logger.info("Loading intent data from intents.json and training.json...")
    
    intents_data = load_json_data(INTENTS_FILE)
    raw_training_data = load_json_data(TRAINING_FILE)

    if not intents_data:
        app_logger.warning(f"Could not load intents from {INTENTS_FILE}. Intent-based responses will not work.")
    if not raw_training_data:
        app_logger.warning(f"Could not load training data from {TRAINING_FILE}. Intent matching will be limited.")

    # Process raw_training_data into training_phrases_map for quick lookup
    if raw_training_data:
        for entry in raw_training_data:
            text = entry.get('text')
            intent = entry.get('intent')
            if text and intent:
                normalized_text = normalize_text(text)
                training_phrases_map[normalized_text] = intent
        app_logger.info(f"Loaded {len(training_phrases_map)} training phrases for intent matching.")

def normalize_text(text):
    """Normalizes text for intent matching (lowercase, remove extra whitespace)."""
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text).strip().lower()

def normalize_url(url):
    """Normalizes a URL for consistent lookup."""
    parsed = urlparse(url)
    # Remove trailing slash unless it's just the root domain
    path = parsed.path.rstrip('/') if parsed.path != '/' else parsed.path
    # Remove common homepage file names
    if path.endswith('/index.html'):
        path = path[:-11].rstrip('/') # Remove /index.html and any trailing slash that results
    elif path.endswith('/index.htm'):
        path = path[:-10].rstrip('/')
    
    clean_url = parsed.scheme + "://" + parsed.netloc.replace('www.', '') + path
    
    # Ensure root domain without path has a trailing slash for consistency if that's the scraper's norm
    if not path and not clean_url.endswith('/'):
        clean_url += '/'
        
    return clean_url

def get_current_time():
    """Returns the current time and approximate location."""
    now = datetime.now()
    current_time_str = now.strftime("%I:%M %p") # e.g., 01:06 PM
    # Assuming the server is located in Coimbatore based on previous context
    return f"The current time is {current_time_str} in Coimbatore, India."

def match_intent(query):
    """
    Attempts to match the user query to a predefined intent using exact or close text matching.
    Returns the intent name if a match is found, otherwise None.
    """
    normalized_query = normalize_text(query)
    app_logger.debug(f"Normalized query for intent matching: '{normalized_query}'")

    # Exact match lookup
    if normalized_query in training_phrases_map:
        matched_intent = training_phrases_map[normalized_query]
        app_logger.info(f"Exact match found for query '{query}' to intent '{matched_intent}'")
        return matched_intent

    # Check for partial matches or keyword presence for robustness (simple example)
    for phrase, intent in training_phrases_map.items():
        if phrase in normalized_query: # User query contains the training phrase
            app_logger.info(f"Partial match found: Query '{query}' contains training phrase '{phrase}' for intent '{intent}'")
            return intent
        elif normalized_query in phrase: # Training phrase contains the user query
            app_logger.info(f"Partial match found: Training phrase '{phrase}' contains query '{query}' for intent '{intent}'")
            return intent

    return None

def get_intent_response(intent_name):
    """Retrieves the response for a given intent."""
    intent_info = intents_data.get(intent_name)
    if not intent_info:
        app_logger.warning(f"No info found for intent: {intent_name}")
        return intents_data.get('fallback', {}).get('response', "Sorry, I couldn't find a response for that.")

    if "response" in intent_info:
        return intent_info["response"]
    elif "response_func" in intent_info:
        func_name = intent_info["response_func"]
        # Correctly call the function based on its name (handle the trailing dot from JSON if present)
        if func_name == "get_current_time" or func_name == "get_current_time.": 
            app_logger.info(f"Calling dynamic response function: {func_name}")
            return get_current_time()
        else:
            app_logger.warning(f"Unknown response function: {func_name} for intent {intent_name}")
            return intents_data.get('fallback', {}).get('response', "Sorry, I couldn't find a response for that.")
    return intents_data.get('fallback', {}).get('response', "Sorry, I couldn't find a response for that.")


def load_or_generate_embeddings():
    """
    Loads pre-computed embeddings or generates them if they don't exist,
    or if the content file has been updated.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls

    # Check if already loaded in this process (e.g., for subsequent requests)
    if model is not None and corpus_embeddings is not None and len(corpus_texts) > 0:
        app_logger.info("Model and embeddings already loaded in this process. Skipping regeneration.")
        return

    app_logger.info("Checking for existing embeddings to load or generate...")

    embeddings_exist_and_not_outdated = False
    if EMBEDDINGS_FILE.exists() and os.path.getsize(EMBEDDINGS_FILE) > 0:
        if CONTENT_FILE.exists() and CONTENT_FILE.stat().st_mtime > EMBEDDINGS_FILE.stat().st_mtime:
            app_logger.warning("Scraped content file is newer than embeddings file. Regenerating embeddings.")
            embeddings_exist_and_not_outdated = False # Force regeneration
        else:
            try:
                with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    corpus_embeddings = torch.tensor(data['embeddings'])
                    corpus_texts = data['texts']
                    corpus_urls = data['urls']
                app_logger.info(f"Loaded {len(corpus_embeddings)} existing embeddings from {EMBEDDINGS_FILE}.")
                embeddings_exist_and_not_outdated = True
            except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError) as e:
                app_logger.warning(f"Error loading embeddings file ({e}). Embeddings will be regenerated.")
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []
            except Exception as e:
                app_logger.warning(f"An unexpected error occurred while loading embeddings ({e}). Embeddings will be regenerated.")
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []

    if embeddings_exist_and_not_outdated:
        if model is None:
            try:
                app_logger.info("Model was None, loading SentenceTransformer model 'all-mpnet-base-v2' after loading embeddings...")
                model = SentenceTransformer('all-mpnet-base-v2')
                app_logger.info("SentenceTransformer model loaded successfully (after embeddings).")
            except Exception as e:
                app_logger.error(f"Failed to load SentenceTransformer model after embeddings: {e}")
                model = None
                corpus_embeddings = None
                corpus_texts = []
                corpus_urls = []
        if model is not None and corpus_embeddings is not None:
            return # Successfully loaded existing embeddings and model.

    app_logger.info("Embeddings not found/loaded or content updated. Generating new embeddings.")

    # Ensure scraped_data is loaded before generating embeddings
    if not scraped_data: # Check global scraped_data, assuming load_scraped_content() was called.
        if not load_scraped_content(): # Attempt to load if not already loaded
            app_logger.error("Cannot generate embeddings: Scraped data is empty or missing after attempting to load.")
            corpus_embeddings = None
            corpus_texts = []
            corpus_urls = []
            return

    texts_for_embedding = []
    urls_for_embedding = []
    seen_snippets = set() # For deduplication during embedding generation

    for url, page_content in scraped_data.items():
        # Process paragraphs
        paragraphs = page_content.get('paragraphs', [])
        for p_text in paragraphs:
            cleaned_p_text = p_text.strip()
            if cleaned_p_text and cleaned_p_text not in seen_snippets:
                texts_for_embedding.append(cleaned_p_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_p_text)

        # Process headings
        headings = page_content.get('headings', [])
        for h_text in headings:
            cleaned_h_text = h_text.strip()
            if cleaned_h_text and cleaned_h_text not in seen_snippets:
                texts_for_embedding.append(cleaned_h_text)
                urls_for_embedding.append(url)
                seen_snippets.add(cleaned_h_text)
        
        # Process specific lists like 'services', 'products', 'use_cases', 'benefits', 'features'
        for category_key in ['services', 'products', 'use_cases', 'benefits', 'features']:
            items = page_content.get(category_key, [])
            if isinstance(items, list): 
                for item_text in items:
                    cleaned_item_text = str(item_text).strip()
                    if cleaned_item_text: 
                        page_title = page_content.get('page_title', url.split('/')[-1].replace('.html', '').replace('-', ' ').title())
                        # This forms the embedding text for structured items like "Service of Homepage: Digital Transformation"
                        text_to_embed = f"{category_key.replace('_', ' ').title()} of {page_title}: {cleaned_item_text}"
                        
                        if text_to_embed not in seen_snippets:
                            texts_for_embedding.append(text_to_embed)
                            urls_for_embedding.append(url)
                            seen_snippets.add(text_to_embed) 

        # Add 'extracted_text' as a last resort or for larger chunks of text
        extracted_full_text = page_content.get('extracted_text', '')
        if extracted_full_text and extracted_full_text.strip() and extracted_full_text.strip() not in seen_snippets:
            texts_for_embedding.append(extracted_full_text.strip())
            urls_for_embedding.append(url)
            seen_snippets.add(extracted_full_text.strip())

    if not texts_for_embedding:
        app_logger.error("Cannot generate embeddings: Extracted texts for embedding are empty after processing JSON structure. Check your scraped_content.json.")
        corpus_embeddings = None
        corpus_texts = []
        corpus_urls = []
        return

    try:
        if model is None:
            app_logger.info("Loading SentenceTransformer model 'all-mpnet-base-v2' for generation...")
            model = SentenceTransformer('all-mpnet-base-v2')
            app_logger.info("SentenceTransformer model loaded successfully.")
        else:
            app_logger.info("SentenceTransformer model already loaded, reusing for generation.")
    except Exception as e:
        app_logger.error(f"Failed to load SentenceTransformer model for generation: {e}")
        model = None 
        return

    app_logger.info(f"Generating embeddings for {len(texts_for_embedding)} texts. This may take a while...")
    try:
        corpus_embeddings = model.encode(texts_for_embedding, convert_to_tensor=True, show_progress_bar=True)
        corpus_texts = texts_for_embedding
        corpus_urls = urls_for_embedding

        embeddings_data = {
            'embeddings': corpus_embeddings.tolist(),
            'texts': corpus_texts,
            'urls': corpus_urls
        }
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=4)
        app_logger.info(f"Embeddings generated and saved to {EMBEDDINGS_FILE}.")
    except Exception as e:
        app_logger.error(f"Error during embedding generation: {e}")
        corpus_embeddings = None
        corpus_texts = []
        corpus_urls = []

def get_specific_list_response(query):
    """
    Checks if the query is asking for a specific list (services, products, use cases, benefits, features).
    If so, retrieves the list from scraped_content.json (index.html or other relevant page) and formats it.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls, homepage_url # Added homepage_url to global

    query_lower = query.lower().strip()
    
    services_keywords = ["services", "list of services", "our services", "what services do you offer", "tell me about your services", "company services"]
    products_keywords = ["products", "list of products", "our products", "what products do you have", "tell me about your products", "company products"]
    use_cases_keywords = ["use case","use cases", "list of use cases", "our use cases", "examples of use cases", "tell me about use cases", "company use cases"] # Added "applications"
    benefits_keywords = ["benefits of", "advantages of", "what are the benefits of", "tell me the benefits of", "benefits", "advantages", "perks", "value proposition"] # Added direct keywords
    features_keywords = ["features of", "functionalities of", "capabilities of", "what are the features of", "tell me the features of", "features", "functionalities", "capabilities"] # Added direct keywords

    target_category = None
    response_heading = ""
    target_topic = None 

    if any(keyword in query_lower for keyword in services_keywords):
        target_category = "services"
        response_heading = "Here are the services offered by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in products_keywords):
        target_category = "products"
        response_heading = "Here are the products offered by KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in use_cases_keywords):
        target_category = "use_cases"
        response_heading = "Here are some use cases for KTG Software and Consulting:"
    elif any(keyword in query_lower for keyword in benefits_keywords):
        target_category = "benefits"
        match = re.search(r'(?:benefits of|advantages of|what are the benefits of|tell me the benefits of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            response_heading = f"Here are some key benefits of {target_topic}:"
        else:
            response_heading = "Here are some key benefits:"
    elif any(keyword in query_lower for keyword in features_keywords):
        target_category = "features"
        match = re.search(r'(?:features of|functionalities of|capabilities of|what are the features of|tell me the features of)\s+(.+)', query_lower)
        if match:
            target_topic = match.group(1).strip()
            target_topic = re.sub(r'(?:your|the|a|an)$', '', target_topic).strip()
            response_heading = f"Here are some key features of {target_topic}:"
        else:
            response_heading = "Here are some key features:"

    if target_category:
        try:
            # Ensure scraped_data is up-to-date before accessing
            # This line is intentionally removed here, as scraped_data is global and loaded once at startup.
            # If load_json_data(CONTENT_FILE) were called here, it would reload for every request, which is inefficient.
            
            if target_category in ["benefits", "features"] and target_topic:
                app_logger.info(f"Attempting to find benefits/features for topic: {target_topic}")
                
                if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
                    load_or_generate_embeddings() # Ensure model and embeddings are loaded

                if model and corpus_embeddings is not None and len(corpus_texts) > 0:
                    topic_query_embedding = model.encode(target_topic, convert_to_tensor=True, show_progress_bar=False)
                    
                    cosine_scores = util.cos_sim(topic_query_embedding, corpus_embeddings)[0]
                    
                    top_k_for_topic = min(len(corpus_texts), 50)
                    top_topic_results = torch.topk(cosine_scores, k=top_k_for_topic)
                    
                    best_url_for_topic = None
                    max_topic_score = -1
                    
                    relevant_urls_found = set()
                    
                    for score, idx in zip(top_topic_results[0], top_topic_results[1]):
                        if score < 0.60:
                            continue 
                        
                        current_url = corpus_urls[idx]
                        relevant_urls_found.add((current_url, score))

                    if relevant_urls_found:
                        url_max_scores = {}
                        for url, score in relevant_urls_found:
                            url_max_scores[url] = max(url_max_scores.get(url, 0), score)
                        
                        sorted_urls = sorted(url_max_scores.items(), key=lambda item: item[1], reverse=True)
                        best_url_for_topic = sorted_urls[0][0]
                        max_topic_score = sorted_urls[0][1]

                    if best_url_for_topic and max_topic_score > 0.5:
                        app_logger.info(f"Most relevant page for '{target_topic}' is {best_url_for_topic} with score {max_topic_score}")
                        page_data = scraped_data.get(best_url_for_topic)
                        if page_data:
                            item_list = page_data.get(target_category, [])
                            if item_list and isinstance(item_list, list):
                                unique_items = []
                                seen_display_items = set()
                                for item in item_list:
                                    cleaned_item = str(item).strip()
                                    if cleaned_item and cleaned_item not in seen_display_items:
                                        unique_items.append(cleaned_item)
                                        seen_display_items.add(cleaned_item)

                                if unique_items:
                                    list_items = "\n".join([f"- {item}" for item in unique_items])
                                    response_text = f"{response_heading}\n{list_items}\n\n(Source: {best_url_for_topic})"
                                    return [('Answer', response_text)]
                                else:
                                    app_logger.warning(f"No explicit '{target_category}' list found for '{target_topic}' on {best_url_for_topic}. Falling back to semantic search.")
                                    return None
                            else:
                                app_logger.warning(f"'{target_category}' field on {best_url_for_topic} is not a list or is empty. Falling back to semantic search.")
                                return None
                        else:
                            app_logger.warning(f"Could not retrieve data for best_url: {best_url_for_topic}. Falling back to semantic search.")
                            return None
                    else:
                        app_logger.info(f"Could not find a highly relevant page for topic '{target_topic}' for direct list retrieval. Falling back to semantic search.")
                        return None
                else:
                    app_logger.error("Model or embeddings not available for specific list retrieval. Falling back to semantic search.")
                    return None
            
            # --- Corrected logic for general categories (services, products, use_cases) ---
            # This part will only execute if it's not a benefits/features query with a specific topic
            # or if the benefits/features direct lookup failed.
            if target_category in ["services", "products", "use_cases"]: 
                # CRITICAL FIX: Use homepage_url instead of hardcoded string
                # Ensure homepage_url is properly set from load_scraped_content
                if not homepage_url:
                    app_logger.error("homepage_url is not set. Cannot retrieve specific lists.")
                    return [('Answer', f"I'm still setting up my knowledge base. Please try again in a moment.")]

                index_page_data = scraped_data.get(homepage_url) 
                if index_page_data:
                    item_list = index_page_data.get(target_category, []) 
                    if item_list and isinstance(item_list, list):
                        list_items = "\n".join([f"- {item}" for item in item_list])
                        response_text = f"{response_heading}\n{list_items}\n\n(Source: {homepage_url})" # Also use homepage_url here
                        return [('Answer', response_text)]
                    else:
                        return [('Answer', f"I couldn't find a dedicated list of {target_category} on the homepage. Please try a more specific question or try semantic search for details.")]
                else: 
                    return [('Answer', f"I couldn't access the homepage data for iTech Software Group to retrieve the list of {target_category}. Please ensure '{homepage_url}' data is present in your scraped_content.json.")]
            
            return None # If it's a benefits/features query but no topic or specific list found, semantic search will run

        except Exception as e:
            app_logger.error(f"Error retrieving specific list for '{target_category}' with topic '{target_topic}': {e}")
            return [('Error', 'An error occurred while trying to fetch the list. Please try again later.')]
    
    return None

# --- Semantic Search Function ---
def search_content(query, max_results=1, max_length=500): 
    """
    Performs a semantic search on the pre-computed embeddings.
    It aims to find the single most relevant and substantial answer.
    """
    global model, corpus_embeddings, corpus_texts, corpus_urls

    if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
        app_logger.warning("Model/Embeddings not fully initialized during search request. Attempting to load/regenerate.")
        load_or_generate_embeddings()

        if model is None or corpus_embeddings is None or len(corpus_texts) == 0:
            app_logger.error("Failed to load/regenerate model or embeddings. Cannot perform search.")
            return [('Error', 'The chatbot is still initializing or encountered an error loading its knowledge. Please try again in a moment.')]

    try:
        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        cosine_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        top_k_candidates = min(len(corpus_texts), 200) 
        top_results = torch.topk(cosine_scores, k=top_k_candidates) 
        
        results_by_url = {} 
        
        sorted_indices = top_results[1][top_results[0].argsort(descending=True)]
        sorted_scores = top_results[0][top_results[0].argsort(descending=True)]

        score_threshold = 0.58 

        query_lower = query.lower()
        is_benefits_query = "benefits of" in query_lower or "advantages of" in query_lower or query_lower == "benefits" or query_lower == "advantages"
        is_features_query = "features of" in query_lower or "functionalities of" in query_lower or "capabilities of" in query_lower or query_lower == "features" or query_lower == "functionalities" or query_lower == "capabilities"

        for score, idx in zip(sorted_scores, sorted_indices):
            if score < score_threshold:
                continue 
            
            original_text = corpus_texts[idx]
            source_url = corpus_urls[idx]

            is_substantial_text = len(original_text.split()) > 10 

            current_best_for_url = results_by_url.get(source_url)

            text_starts_with_benefits_prefix = original_text.lower().startswith("benefits of")
            text_starts_with_features_prefix = original_text.lower().startswith("features of")

            should_prioritize_this_snippet = False
            if is_benefits_query and text_starts_with_benefits_prefix and score > 0.65:
                should_prioritize_this_snippet = True
            elif is_features_query and text_starts_with_features_prefix and score > 0.65:
                should_prioritize_this_snippet = True

            if should_prioritize_this_snippet:
                results_by_url[source_url] = {
                    'score': score,
                    'text': original_text,
                    'url': source_url,
                    'is_substantial': True 
                }
            elif current_best_for_url is None:
                results_by_url[source_url] = {
                    'score': score,
                    'text': original_text,
                    'url': source_url,
                    'is_substantial': is_substantial_text
                }
            else:
                if is_substantial_text and not current_best_for_url['is_substantial']:
                    results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }
                elif score > current_best_for_url['score']:
                    results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }
                elif (score == current_best_for_url['score'] and
                      is_substantial_text == current_best_for_url['is_substantial'] and
                      len(original_text) > len(current_best_for_url['text'])):
                     results_by_url[source_url] = {
                        'score': score,
                        'text': original_text,
                        'url': source_url,
                        'is_substantial': is_substantial_text
                    }

        final_results = []
        sorted_best_snippets = sorted(
            results_by_url.values(), 
            key=lambda x: (x['is_substantial'], x['score'], len(x['text'])),
            reverse=True 
        )
        
        if sorted_best_snippets:
            best_snippet_info = sorted_best_snippets[0]

            is_generic_short_answer_to_definition_query = (
                ("what is" in query_lower or "what are" in query_lower or "define" in query_lower or "explain" in query_lower) and
                len(best_snippet_info['text'].split()) < 10 and 
                best_snippet_info['score'] > 0.75 and 
                best_snippet_info['text'].lower().strip() in query_lower.strip() and
                not best_snippet_info['is_substantial']
            )
            
            if is_generic_short_answer_to_definition_query:
                return [('Answer', 'I could not find a detailed definition or explanation for that specific term in the scraped content. Please try rephrasing your question or asking for specific aspects.')]

            if (is_benefits_query and best_snippet_info['text'].lower().startswith("benefits of")) or \
               (is_features_query and best_snippet_info['text'].lower().startswith("features of")):
                
                prefix_removed_text = re.sub(r'^(benefits of|features of).*?:', '', best_snippet_info['text'], flags=re.IGNORECASE).strip()
                
                final_results.append(('Answer', f"Here's a key point:\n- {prefix_removed_text} (Source: {best_snippet_info['url']})"))

            else:
                short_answer = best_snippet_info['text'][:max_length] 
                if len(best_snippet_info['text']) > max_length:
                    last_period_idx = short_answer.rfind('.')
                    if last_period_idx > max_length * 0.8: 
                        short_answer = short_answer[:last_period_idx + 1]
                    else:
                        short_answer += '...'
                final_results.append(('Answer', f"{short_answer} (Source: {best_snippet_info['url']})"))

        if not final_results:
            return [('Answer', 'I could not find a relevant answer in the scraped content. Please try rephrasing your question.')]
        
        return final_results

    except Exception as e:
        app_logger.error(f"Error during semantic search for query '{query}': {e}")
        return [('Error', 'An error occurred during the search. Please try again later.')]

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    app_logger.info("Received request to /search endpoint.")
    
    data = request.get_json(silent=True, force=True) 

    app_logger.info(f"Request data received by Flask: {data}")

    if data is None:
        app_logger.error("Request body is not valid JSON or is empty, or Content-Type is incorrect.")
        return jsonify({'response': 'Error: Invalid or empty request body. Please ensure your query is sent as valid JSON (check browser console for network errors).'}), 400

    user_query = data.get('query')

    app_logger.info(f"Extracted query: '{user_query}'")

    if not user_query:
        app_logger.warning("User query is empty after extraction.")
        return jsonify({'response': 'Please enter a query.'})

    # --- NEW: Prioritize Intent Matching ---
    matched_intent_name = match_intent(user_query)
    if matched_intent_name:
        app_logger.info(f"User query '{user_query}' matched intent '{matched_intent_name}'.")
        response_from_intent = get_intent_response(matched_intent_name)
        return jsonify({'response': response_from_intent})
    
    # --- Existing Logic: Specific List Requests ---
    list_response = get_specific_list_response(user_query)
    if list_response:
        app_logger.info(f"Sending specific list response: {list_response[0][1][:100]}...")
        return jsonify({'response': list_response[0][1]}) 

    # --- Existing Logic: Semantic Search as Fallback ---
    search_results = search_content(user_query)
    
    # Format the response for the frontend
    formatted_responses = [f"{content}" for type_, content in search_results]
    
    response_text = "\n\n".join(formatted_responses)
    app_logger.info(f"Sending response: {response_text[:100]}...")

    return jsonify({'response': response_text})

# --- Main execution block ---
if __name__ == '__main__':
    app_logger.info("Starting Flask application...")
    # Load scraped content FIRST, as homepage_url is needed for other loads
    if not load_scraped_content():
        app_logger.critical("Application cannot start without scraped content. Exiting.")
        # Optionally exit or raise an error here if content is mandatory for app to run
        # For development, you might continue but expect errors for content-dependent features
    
    load_intent_data() # Load intents and training data on startup
    load_or_generate_embeddings() # Load/generate scraped content embeddings on startup (this now correctly uses global scraped_data)
    
    app.run(debug=True)
