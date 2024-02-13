from transformers import AutoTokenizer, OPTForQuestionAnswering, pipeline
import torch

model = OPTForQuestionAnswering.from_pretrained('C:/Users/Joshua/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c')
tokenizer = AutoTokenizer.from_pretrained('C:/Users/Joshua/.cache/huggingface/hub/models--facebook--opt-350m/snapshots/08ab08cc4b72ff5593870b5d527cf4230323703c')

'''prompt = 'Generate 5 synonymous sentences for the instruction "Derain the image"'
text = 'Remove the rain from the image'
inputs = tokenizer(prompt, text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
answer_offset = len(tokenizer(prompt)[0])

predict_answer_tokens = inputs.input_ids[
    0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
]

predicted = tokenizer.decode(predict_answer_tokens)
print(predicted)'''

generator = pipeline('text-generation', model='C:/Users/Joshua/.cache/huggingface/hub/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254', do_sample=True)

print(generator("Generate a synonymous prompt to the instruction 'Derain the image'"))