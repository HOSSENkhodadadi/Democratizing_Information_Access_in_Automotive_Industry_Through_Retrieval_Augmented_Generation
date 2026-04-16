from flask import Flask, render_template, request, redirect, url_for
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer
import random

app = Flask(__name__)

MODEL_PATH_ROOT = "C:\\Users\\hossein.khodadadi\\OneDrive - JATO Dynamics Ltd\\Desktop\\table_question_answering\\models\\"
GOLD_CSV_PATH = "gold_initial.csv"
NLQ_DIRECTORY = "query_dataset_10_255_main.json"
VEC_DIM = 768 # IT CAN ALSO BE 384
SE_Cap = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize global variables
semantic_searcher = None
new_row_strings_list = None
car_spec_NQ = None
transformer = None

# There was a problem with loading the paraphrase-multilingual-MiniLM-L12-v2
models_list = [
    {"model_name": "paraphrase-multilingual-mpnet-base-v2", "embedding_dimension": 768},
    {"model_name": "paraphrase-multilingual-MiniLM-L12-v2", "embedding_dimension": 384},
    {"model_name": "multi-qa-mpnet-base-dot-v1", "embedding_dimension": 768},
    {"model_name": "multi-qa-MiniLM-L6-cos-v1", "embedding_dimension": 384},
    {"model_name": "multi-qa-distilbert-cos-v1", "embedding_dimension": 768},
    {"model_name": "all-mpnet-base-v2", "embedding_dimension": 768},
    {"model_name": "all-MiniLM-L12-v2", "embedding_dimension": 384},
    {"model_name": "all-MiniLM-L6-v2", "embedding_dimension": 384},
    {"model_name": "distiluse-base-multilingual-cased-v1", "embedding_dimension": 512},
    {"model_name": "distiluse-base-multilingual-cased-v2", "embedding_dimension": 512},
    {"model_name": "paraphrase-albert-small-v2", "embedding_dimension": 768, 'status': 'unscanned'},
    {"model_name": "all-distilroberta-v1", "embedding_dimension": 768, 'status': 'unscanned'},
    {"model_name": "paraphrase-MiniLM-L3-v2", "embedding_dimension": 384, 'status': 'unscanned'},
]

model_directories = []
for item in models_list:
    model_path = MODEL_PATH_ROOT + item['model_name']
    direc_dim = {'directory':model_path,'dim': item["embedding_dimension"]}
    model_directories.append(direc_dim)

# print(model_directories)

# print(model_directories[0]['directory'].split("\\")[7])

# Loading the carspecs

gold_df = pd.read_csv(GOLD_CSV_PATH)
row_string = gold_df.iloc[50].to_string()
row_list = []
for index, row in gold_df.iterrows():
    if index >= 200:
        break
    row_list.append(row.to_string())
# print(row_list[0])


# Cleaning the car spec dataset <br>
# Data cleaning and preprocessing including:<br> 1. Removing nan variables<br> 2. removing the addtional coloumns <br>3. restructuring the sentence

def reorder_remove_nan(row):
    row_string = []
    for col in gold_df.columns:
        if col != 'Unnamed: 0' and pd.notnull(row[col]):
            row_string.append(f"{col} is {row[col]}")
    return 'For this car the ' + ', '.join(row_string)

def initialize_data():
    global semantic_searcher, new_row_strings_list, car_spec_NQ, transformer
    
    # Load and process the data
    gold_df = pd.read_csv(GOLD_CSV_PATH)
    new_row_strings = gold_df.apply(reorder_remove_nan, axis=1)
    new_row_strings_list = new_row_strings.to_list()
    
    with open(NLQ_DIRECTORY, 'r') as file:
        car_spec_NQ = json.load(file)
    
    # Initialize the encoder and search engine
    print('Encoding the sentences...')
    transformer = RowTransformer(model_directories[5]['directory'])
    embeddings = []
    for sentence in tqdm(new_row_strings_list, desc="Encoding sentences"):
        embedding = transformer.encode_sentence(sentence)
        embeddings.append(embedding)
    
    # Stack embeddings into a single tensor
    embeddings = torch.stack(embeddings).squeeze()
    embeddings = embeddings.numpy()  
    
    print('Initializing the search engine')
    semantic_searcher = SearchEngine('faiss', 'CS', SE_Cap, model_directories[5]['dim'])
    print('Indexing the embeddings')
    semantic_searcher.Index(embeddings=embeddings)

class RowTransformer:
    def __init__(self, model_path = model_directories[5]['directory']):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        # First move to empty device, then to target device
        # self.model = self.model.to_empty(device=DEVICE)
        self.model = self.model.to(DEVICE)

    def encode_sentence(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings

class SearchEngine:
    def __init__(self, name, sim_metric, capacity = SE_Cap, vec_dim = model_directories[5]['dim']):
        self.name = name
        self.sim_metric = sim_metric
        self.capacity = capacity
        self.vec_dim = vec_dim
        if self.sim_metric == 'L2':
            self.engine = faiss.IndexFlatL2(self.vec_dim)
        elif self.sim_metric in ['IP', 'CS']:
            self.engine = faiss.IndexFlatIP(self.vec_dim)
    def Index(self,embeddings: torch.Tensor) -> None:
        if self.sim_metric in ['L2','IP']:
            self.engine.add(embeddings)
        elif self.sim_metric == 'CS':
            normalized_passage_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.engine.add(normalized_passage_embeddings)
    def Search(self, embedding: torch.Tensor)-> tuple:
        # Move tensor to CPU and convert to numpy
        if self.sim_metric == 'L2':
            distances, indices = self.engine.search(embedding, self.capacity)
        elif self.sim_metric == 'IP':
            distances, indices = self.engine.search(embedding, self.capacity)
        elif self.sim_metric == 'CS':
            normalized_query_embeddings = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            distances, indices = self.engine.search(normalized_query_embeddings, self.capacity)
        return distances, indices
    
@app.route('/')
def index():
    rand_q_indices = random.sample(range(len(car_spec_NQ)), 5)
    rnd_query_list = [
        {"index": idx, "Question": car_spec_NQ[idx]["Question"]}
        for idx in rand_q_indices
    ]
    return render_template('index.html', queries=rnd_query_list)

@app.route('/process_query', methods=['GET', 'POST'])
def process_query():
    if request.method == 'GET':
        index = int(request.args.get('index', -1))
        print('The returned by front is: ', index)
        if index >= 0:
            returned_text = car_spec_NQ[index]['Question']
            correct_rows = car_spec_NQ[index]['Correct_rows']
        else:
            return redirect(url_for('index'))
    else:
        returned_text = request.form.get('customQuery', '')
        correct_rows = []
    
    # Process the query
    question_embedding = transformer.encode_sentence(returned_text)
    question_embedding = question_embedding.numpy().reshape(1, -1)  # Ensure 2D numpy array
    _, indices = semantic_searcher.Search(question_embedding)
    
    # Prepare retrieved items
    retrieved_items = [
        {'index': int(idx), 'text': new_row_strings_list[int(idx)]}
        for idx in indices[0][:5]
    ]
    
    # Prepare related items
    related_items = [
        {'index': idx, 'text': new_row_strings_list[idx]}
        for idx in correct_rows
    ]
    
    # Prepare context for LLM
    context = []
    for idx in indices[0][:5]:
        context.append(f"{idx}: {new_row_strings_list[int(idx)]}")
    str_context = ';'.join(context)
    
    context = []
    for idx in indices[0][:5]:
        context.append(f"{idx}: {new_row_strings_list[int(idx)]}")
    str_context = ';'.join(context)
    
    # Get LLM response (placeholder for now)

    from generator import Generator
    question = returned_text
    prompt = f"Respond this query: {question}, based on the following context: {str_context}"
    print(prompt)
    llm_response = Generator().prompt_meta(prompt = prompt)
    
    return render_template(
        'results.html',
        query_text=returned_text,
        retrieved_items=retrieved_items,
        related_items=related_items,
        llm_response=llm_response
    )

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, port=5005)