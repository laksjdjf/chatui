import gradio as gr
from tabs.setting import generate, get_prompt
import faiss
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

pre_prompt = ""
stop_generate = False

default_template = "以下の付加情報を元に、質問に答えてください。\n{augment}\n質問:{input}"

retrieval_model = None

class FaissTextRetrieval:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.passage_texts = []
        self.index = None
        self.device = "cpu"
        
    def to(self, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype if device == "cuda:0" else torch.float32
        self.model.to(device, dtype=dtype)
    
    @torch.no_grad()
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    @torch.no_grad()
    def get_embeddings(self, input_texts: list) -> Tensor:
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        input_ids = batch_dict["input_ids"].to(self.device)
        attention_mask = batch_dict["attention_mask"].to(self.device)
        
        # Get model outputs
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply average pooling
        embeddings = self.average_pool(outputs.last_hidden_state, attention_mask)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    @torch.no_grad()
    def add_passages(self, passages: list, batch_size: int=16):
        self.passage_texts.extend(passages)
        passages = ["passage:" + passage for passage in passages]
        embeddings = []
        print("Computing embeddings...")
        for i in tqdm(range(0, len(passages), batch_size)):
            embeddings.append(self.get_embeddings(passages[i:i+batch_size]))
        embeddings = torch.cat(embeddings).float().cpu().numpy()

        if self.index is None:
            d = embeddings.shape[1]  # Dimension of embeddings
            self.index = faiss.IndexFlatL2(d)  # Create a faiss index
            self.index.add(embeddings)  # Add passages to the index
        else:
            self.index.add(embeddings)  # Add passages to the existing index
    
    def search(self, queries: list, top_k: int = 5) -> list:
        queries = ["query:" + query for query in queries]
        query_embeddings = self.get_embeddings(queries).float().cpu().numpy()
        distances, indices = self.index.search(query_embeddings, top_k)

        results = []
        for i in range(len(queries)):
            result = [(self.passage_texts[idx], distances[i][j]) for j, idx in enumerate(indices[i])]
            results.append(result)
        return results
    
    def reset(self):
        self.passage_texts = []
        self.index = None

def chunk_text(file_path: str, chunk_size: int = 256, overlap: int = 20) -> list:
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    chunks = []
    current_pos = 0
    
    text_length = len(text)
    
    while current_pos < text_length:
        end_pos = current_pos + chunk_size
        
        if end_pos > text_length:
            end_pos = text_length
        
        chunks.append(text[current_pos:end_pos])
        current_pos += chunk_size - overlap
    
    return chunks

def load_retrieval_model(model_name, device, dtype, file, chunk_size, overlap_size):
    global retrieval_model
    retrieval_model = FaissTextRetrieval(model_name)
    dtype = torch.float32 if dtype == "float32" else torch.float16
    retrieval_model.to(device, dtype)

    passages = chunk_text(file, chunk_size, overlap_size)
    retrieval_model.add_passages(passages)
    return "Loaded"

def stop():
    global stop_generate
    stop_generate = True

def rag_handler(input, template, num_aug):
    global stop_generate
    stop_generate = False

    rag_results = retrieval_model.search([input], top_k=num_aug)

    augment = "\n".join([aug[0] for aug in rag_results[0]])
    prompt = template.format(augment=augment, input=input)
    prompt = get_prompt(prompt)

    output = ""
    yield gr.update(value=output), gr.update(value=prompt), gr.update(visible=False), gr.update(visible=True)
    for text in generate(prompt):
        if stop_generate:
            break

        output += text
        yield gr.update(value=output), gr.update(value=prompt), gr.update(visible=False), gr.update(visible=True)

    yield gr.update(value=output), gr.update(value=prompt), gr.update(visible=True), gr.update(visible=False)


def rag():
    with gr.Blocks() as rag_interface:
        gr.Markdown("なんちゃってRAG用タブです。")
        input_textbox = gr.Textbox(
            label="Input",
            placeholder="Enter your input here...",
            interactive=True,
            lines=3,
        )

        num_aug = gr.Slider(label="Number of Augmentations", minimum=1, maximum=5, step=1, value=1)

        output_textbox = gr.Textbox(
            label="Output",
            interactive=False,
            lines=3,
        )

        generate_button = gr.Button("Generate", variant="primary")
        stop_button = gr.Button("Stop", visible=False)
        
        rag_template = gr.Textbox(
            label="Template",
            value=default_template,
            interactive=True,
            lines=3,
        )
        
        prompt_text = gr.Textbox(
            label="Prompt Text",
            interactive=False,
            lines=3,
        )

        with gr.Accordion("Embedding Model"):
            model_name = gr.Textbox(label="Model Name", value="intfloat/multilingual-e5-large", lines=1)
            device = gr.Dropdown(label="Device", choices=["cpu", "cuda:0"], value="cuda:0")
            dtype = gr.Dropdown(label="Data Type", choices=["float32", "float16"], value="float16")
            file = gr.File(label="Text File", type="filepath")
            chunk_size = gr.Slider(label="Chunk Size", value=512, minimum=1, maximum=1024, step=1)
            overlap_size = gr.Slider(label="Overlap Size", value=20, minimum=0, maximum=512, step=1)
            load_button = gr.Button("Load")
            load_result = gr.Textbox(label="Load Result", value="", lines=1)


        input_textbox.submit(
            rag_handler,
            inputs=[input_textbox, rag_template, num_aug],
            outputs=[output_textbox, prompt_text, generate_button, stop_button],
        )

        generate_button.click(
            rag_handler,
            inputs=[input_textbox, rag_template, num_aug],
            outputs=[output_textbox, prompt_text, generate_button, stop_button],
        )

        stop_button.click(stop)

        load_button.click(
            load_retrieval_model,
            inputs=[model_name, device, dtype, file, chunk_size, overlap_size],
            outputs=[load_result],
        )

    return rag_interface
