import torch, random, argparse
from PIL import ImageDraw, Image
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

'''parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('model')
args = parser.parse_args()'''

def transform(example_batch):
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

def show_examples_with_classification(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

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

            inputs = processor(images=image, return_tensors="pt")
            predictions = model(**inputs)
            logits = predictions.logits
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            predicted_label = labels[predictions.item()]

            draw.text((box[0] + 10, box[1]), f"Ground truth: {label}", fill=(0, 0, 0))
            draw.text((box[0] + 10, box[1] + 20), f"Prediction: {predicted_label}", fill=(0, 0, 0))

    return grid


dataset = load_dataset("imagefolder", data_dir='./material/Nike_Adidas_converse_Shoes_image_dataset', split='test')
labels = dataset.features['label'].names
model_path = './vit-base-shoes/checkpoint-400'
model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
processor = ViTImageProcessor.from_pretrained(model_path)
show_examples_with_classification(dataset, seed=random.randint(0, 3339), examples_per_class=5).save(fp='./eval_3.png')