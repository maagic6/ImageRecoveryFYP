from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch

dataset = load_dataset(...)
# dataset class tbd
class ImageDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __getitem__(self, index):
    item = self.dataset[index]
    encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
    encoding = {k: v.squeeze() for k, v in encoding.items()}
    encoding["text"] = item["text"]
    return encoding

def collate_fn(batch):
  processed_batch = {}
  for key in batch[0].keys():
    if key is not "text":
      processed_batch[key] = torch.stack([example[key]] for example in batch)
    else:
      text_inputs = processor.tokenizer([example["text"] for example in batch], padding=True, return_tensors="pt")
      processed_batch["input_ids"] = text_inputs["input_ids"]
      processed_batch["attention_mask"] = text_inputs["attention_mask"]
  return processed_batch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip-opt2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)
config = LoraConfig(
  r=16,
  lora_alpha=32,
  lora_dropout=0.05,
  bias="none",
  target_modules=["q_proj","k_proj"]
)

model = get_peft_model(model, config)
train_dataset = ImageDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=3, collate_fn=collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.train()

for epoch in range(200):
  print("Epoch:")
  for index, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device, torch.float16)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
    loss = outputs.loss
    print("Loss:", loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()