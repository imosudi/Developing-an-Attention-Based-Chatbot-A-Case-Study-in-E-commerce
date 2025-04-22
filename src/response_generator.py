from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = TFGPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(prompt):
    input_ids = gpt_tokenizer.encode(prompt, return_tensors='tf')
    output = gpt_model.generate(input_ids, max_length=100)
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)