import re
import requests
import torch

# request headers
headers = {
    'Accept-Encoding': 'utf-8',
    'accept': 'application/json',
    'Access-Control-Allow-Credentials': 'true',
    'Content-Type': 'application/x-www-form-urlencoded'
}

def create_embedding(list_of_str, t, m):
    tokens = {
        'input_ids': [],
        'attention_mask': []
    }

    for text in list_of_str:
        new_tokens = t.encode_plus(text, max_length=32, truncation=True,
                                    padding='max_length', return_tensors='pt').to('cuda')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])    

    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = m(**tokens)
    embeddings = outputs.logits
    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled.cpu().detach().numpy()

def translate_text(q):
    translate_params = {
        'q': q, 
        'source': 'en', 
        'target': 'ru', 
        'format': 'text',
        'api_key': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
    }
    translate_api = 'http://host.docker.internal:8080/translate'
    words = q.split(' ')

    for i, word in enumerate(words):
        if re.match('[A-Za-z]', word):
            translate_params['q'] = word
            translated_text = requests.post(
                translate_api, params=translate_params, 
                headers=headers
            ).json()
            try:
                words[i] = translated_text['translatedText']
            except:
                print(f'[ERR] [TRANSLATE] {translated_text}')
    
    return ' '.join(words)

def generate_response(pos, model, tokenizer, cuda_available, use_translation = True):
  instruction = f'Напиши краткое описание кандидата на позицию {pos}'
  bos_token = '<start_of_turn>'
  eos_token = '<end_of_turn>'

  prompt = ""
  prompt += bos_token + 'user\n'
  prompt += instruction
  prompt += eos_token + '\n'
  prompt += bos_token + 'model\n'

  encoded_input = tokenizer(
      prompt, 
      return_tensors="pt", 
      add_special_tokens=True
  )
  model_inputs = encoded_input
  if cuda_available:
    model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)
  answer = decoded_output[0].replace(prompt, "").strip('<bos>')
  answer = answer.split('Технологические навыки:')
  if use_translation:
    answer[0] = translate_text(answer[0]) # translating about-me
  skills = list(set([re.sub("(\s\n|\n)+","",x) for x in answer[1].split('*')]))
  skills = list(filter(lambda skill: len(skill) > 0, skills))
  skills[0] = f'\n* {skills[0]}'
  answer[1] = "\n* ".join(skills)
  answer = 'Технологические навыки:'.join(answer)

  return answer