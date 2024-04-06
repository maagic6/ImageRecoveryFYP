import torch
import numpy as np
from datasets import load_metric, load_dataset
from PIL import ImageDraw, Image
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer

path = './material/Nike_Adidas_converse_Shoes_image_dataset'
dataset = load_dataset("imagefolder", data_dir=path, split='train')
dataset_test = load_dataset("imagefolder", data_dir=path, split='test')

def show_labels():
  unique_labels = set()
  for i, image in enumerate(dataset):
      unique_labels.add(image['label'])
  return unique_labels

def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds.features['label'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)

    for label_id, label in enumerate(labels):
        ds_slice = ds.filter(lambda ex: ex['label'] == label_id).shuffle(seed).select(range(examples_per_class))

        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255))

    return grid

def transform(example_batch):
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    print(example_batch['label'])
    print('wat duh hell oh my gad')
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

#show_examples(dataset, seed=random.randint(0, 1234), examples_per_class=3).save(fp='./test.png')

model_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_path)
processed_dataset = dataset.with_transform(transform)
processed_test_dataset = dataset_test.with_transform(transform)
metric = load_metric("accuracy")
labels = dataset.features['label'].names

model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
  output_dir="./vit-base-finedtuned-shoes",
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=5,
  fp16=False,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=processed_dataset,
    eval_dataset=processed_test_dataset,
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)