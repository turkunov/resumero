import streamlit as st
import os
from utils import generate_response, create_embedding
import torch
import time
import numpy as np
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json

st.set_page_config(
    page_title='Gemma finetuned on HH.RU',
    page_icon='💼'
)

os.environ["HF_READ_TOKEN"] = "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"
os.environ["HF_WRITE_TOKEN"] = "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"

if 'sesh_init' not in st.session_state.keys():
    st.toast('Loading prerequisites for LLM..')

@st.cache_resource
def load_cache():
    torch.cuda.empty_cache()
    adapters_path = './gemma-analytics-resumes-adapter'
    model_name = "google/gemma-2b-it"
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=nf4_config,
        token=os.environ['HF_READ_TOKEN'],
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    m = PeftModel.from_pretrained(m, adapters_path)
    m = m.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ['HF_READ_TOKEN'], padding_side="right", padding='max_length', max_length=256, truncation=True
    )
    with open('experts_config.json', 'r', encoding='windows-1251') as conf:
        config = json.loads(conf.read())
        positions = list(config.keys())

    position_embeddings = create_embedding(positions, tokenizer, m)
    return m, tokenizer, config, positions, position_embeddings

m, tokenizer, conf, pos, pos_embeddings = load_cache()

if 'sesh_init' not in st.session_state.keys():
    st.toast('LLM loaded to cache and relevant vacancies pulled', icon='✅')

st.session_state['sesh_init'] = 1

def main():
    torch.cuda.empty_cache()
    left_column, right_column = st.columns(2)
    vacancy_type = left_column.text_area('Type of vacancy', value='Аналитик на Python')
    cuda_available = right_column.checkbox('CUDA is available')
    debug_enabled = right_column.checkbox('Debug mode')
    generate_btn = right_column.button('Generate')

    if generate_btn:
        st.toast('Analyzing your request..', icon='🤔')
        time.sleep(2)
        vacancy_embedding = create_embedding(
            [vacancy_type], tokenizer, m
        )
        closest_pos = pos[np.argmax(cosine_similarity(
            vacancy_embedding,
            pos_embeddings
        ))]
        print(f'[LOG] Closest position: {closest_pos}')
        st.toast('Resume is being generated..', icon='✨')
        st.code(
            generate_response(closest_pos, m, tokenizer, 
                            cuda_available, not debug_enabled)
        )
        st.toast('Resume generated, have a look!', icon='👌')

if __name__ == '__main__':
    main()