# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer
from transformers import GPT2Model

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:
# 
# `%pip install transformers==4.9.1`



# %pip install transformers==4.9.1






d = {
    'torch': '1.9.0',
    'transformers': '4.9.1',
}
check_packages(d)





# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
    print(f'torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}')
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    get_ipython().system('nvidia-smi')


# # Chapter 16: Transformers – Improving Natural Language Processing with Attention Mechanisms (Part 2/3)

# **Outline**
# 
# - [Building large-scale language models by leveraging unlabeled data](#Building-large-scale-language-models-by-leveraging-unlabeled-data)
#   - [Pre-training and fine-tuning transformer models](#Pre-training-and-fine-tuning-transformer-models)
#   - [Leveraging unlabeled data with GPT](#Leveraging-unlabeled-data-with-GPT)
#   - [Using GPT-2 to generate new text](#Using-GPT-2-to-generate-new-text)
#   - [Bidirectional pre-training with BERT](#Bidirectional-pre-training-with-BERT)
#   - [The best of both worlds: BART](#The-best-of-both-worlds-BART)





# ## Building large-scale language models by leveraging unlabeled data
# ##  Pre-training and fine-tuning transformer models
# 
# 





# ## Leveraging unlabeled data with GPT









# ### Using GPT-2 to generate new text



# %pip install --upgrade torchaudio torch






generator = pipeline('text-generation', model='gpt2')
set_seed(123)
generator("Hey readers, today is",
          max_length=20,
          num_return_sequences=3)





tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "Let us encode this sentence"
encoded_input = tokenizer(text, return_tensors='pt')
encoded_input




model = GPT2Model.from_pretrained('gpt2')




output = model(**encoded_input)
output['last_hidden_state'].shape


# ### Bidirectional pre-training with BERT
# 













# ### The best of both worlds: BART





# ---
# 
# Readers may ignore the next cell.









