from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def generate_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")  # Switch to a larger model
    model = GPT2LMHeadModel.from_pretrained("gpt2-large").cuda()

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    outputs = model.generate(
        inputs,
        max_length=100,        # Limit the length of the output
        temperature=0.7,       # Reduce randomness for more focused output
        top_p=0.9,             # Nucleus sampling
        do_sample=True,        # Enable sampling
        repetition_penalty=1.7, # Reduce repetition
        eos_token_id=tokenizer.eos_token_id  # Use eos_token_id for padding
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(generate_response("What is DevOps?"))
