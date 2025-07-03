# Nemotron Nano VL - FiftyOne Remote Source Zoo Model

NVIDIA's Llama-3.1-Nemotron-Nano-VL-8B-V1 integrated as a Remote Source Zoo Model for FiftyOne, enabling seamless computer vision tasks including object detection, OCR, classification, and visual question answering.

<img src="nemotronvl-lq.gif">

## Installation

```bash
pip install fiftyone transformers accelerate timm einops open-clip-torch torch
```

## Quick Start

### Model Registration & Loading

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Register the model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/Nemotron_Nano_VL", 
    overwrite=True
)

# Load the model
model = foz.load_zoo_model(
    "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    # install_requirements=True # you can pass this if you're not sure if you have all the requirements installed
)
```

### Basic Usage

```python
# Load a dataset
dataset = foz.load_zoo_dataset("quickstart")

# Visual Question Answering
model.operation = "vqa"
model.prompt = "Describe this image"
dataset.apply_model(model, label_field="descriptions")

# Object Detection
model.operation = "detect"
model.prompt = "Find all people and objects"
dataset.apply_model(model, label_field="detections")

# OCR
model.operation = "ocr"
model.prompt = "Extract all text from this image"
dataset.apply_model(model, label_field="text_detections")

# Classification
model.operation = "classify"
model.prompt = "Classify this image: indoor, outdoor, people, animals"
dataset.apply_model(model, label_field="classifications")
```

## Operations

### 1. Visual Question Answering (VQA)

Generate natural language descriptions and answer questions about images.

```python
model.operation = "vqa"
model.prompt = "What is the main subject of this image?"
dataset.apply_model(model, label_field="vqa_results")
```

**Output**: Natural language text response

### 2. Object Detection

Detect and localize objects, people, UI elements with bounding boxes.

```python
model.operation = "detect"
model.prompt = "Find all cars and pedestrians"
dataset.apply_model(model, label_field="detections")
```

**Output**: `fiftyone.Detections` with normalized bounding boxes

### 3. Optical Character Recognition (OCR)

Extract and localize text from images, documents, and user interfaces.

```python
model.operation = "ocr"
model.prompt = "Extract all text from this document"
dataset.apply_model(model, label_field="ocr_results")
```

**Output**: `fiftyone.Detections` with text content and bounding boxes

### 4. Image Classification

Classify images into multiple categories.

```python
model.operation = "classify"
model.prompt = "Classify: indoor/outdoor, day/night, urban/rural"
dataset.apply_model(model, label_field="classifications")
```

**Output**: `fiftyone.Classifications` with multiple labels

## Advanced Workflows

### Context-Aware Processing

Use VQA results to guide subsequent operations:

```python
# Generate detailed descriptions
model.operation = "vqa"
model.prompt = "Describe this image in detail"
dataset.apply_model(model, label_field="vqa_results")

# Use descriptions to guide detection
model.operation = "detect"
dataset.apply_model(model, prompt_field="vqa_results", label_field="context_detections")
```

### Smart OCR with Fallback

```python
model.operation = "ocr"
model.prompt = "Extract all text. If no text exists, return 'no text' with a bounding box around the whole image."
dataset.apply_model(model, label_field="ocr_results")
```

### Custom System Prompts

Override default system prompts for specialized tasks:

```python
model.system_prompt = """You are a specialized assistant for detecting vehicles.
Focus only on cars, trucks, and motorcycles. Return results as JSON."""

model.operation = "detect"
model.prompt = "Find all vehicles"
dataset.apply_model(model, label_field="vehicle_detections")
```

## Evaluation

### Evaluating Detections

Use FiftyOne's evaluation API to assess detection performance against ground truth:

```python
# Evaluate predicted detections against ground truth
results = dataset.evaluate_detections(
    pred_field="detections",           # Field containing predictions
    gt_field="ground_truth",           # Field containing ground truth
    eval_key="nemotron_eval",          # Unique key for this evaluation
    iou=0.5,                          # IoU threshold for matches
    classwise=True                     # Match objects with same class only
)

# Print evaluation metrics
print(results.mAP())

# View detailed results
results.print_report()

# Create plots
plot = results.plot_confusion_matrix()
plot.show()

plot = results.plot_pr_curves()
plot.show()
```

### Evaluation Options

```python
# Class-specific evaluation
results = dataset.evaluate_detections(
    pred_field="detections",
    gt_field="ground_truth", 
    eval_key="class_eval",
    classes=["person", "car", "bicycle"],  # Specific classes to evaluate
    iou=0.75,                             # Stricter IoU threshold
    classwise=True
)

# Cross-class evaluation (allow matches between different classes)
results = dataset.evaluate_detections(
    pred_field="detections",
    gt_field="ground_truth",
    eval_key="cross_class_eval", 
    classwise=False
)
```

### Viewing Evaluation Results

After evaluation, samples are annotated with TP/FP/FN information:

```python
# View samples with evaluation results
session = fo.launch_app(dataset)

# Filter to false positives
fp_view = dataset.filter_labels("detections", F("nemotron_eval") == "fp")

# Filter to false negatives  
fn_view = dataset.filter_labels("ground_truth", F("nemotron_eval") == "fn")

# View high-confidence false positives
high_conf_fp = dataset.filter_labels(
    "detections", 
    (F("nemotron_eval") == "fp") & (F("confidence") > 0.8)
)
```

## Use Cases

### Document Intelligence

```python
# Extract document structure
model.operation = "ocr"
model.prompt = "Extract all text and categorize by type: title, paragraph, caption, table"
dataset.apply_model(model, label_field="document_structure")

# Summarize document content
model.operation = "vqa"
model.prompt = "Summarize the main content and key findings"
dataset.apply_model(model, label_field="document_summary")
```

### UI Analysis

```python
# Detect interface elements
model.operation = "detect"
model.prompt = "Find all buttons, text fields, menus, and interactive elements"
dataset.apply_model(model, label_field="ui_elements")

# Extract UI text
model.operation = "ocr"
model.prompt = "Extract all text from buttons, labels, and menus"
dataset.apply_model(model, label_field="ui_text")
```

### Multi-Modal Analysis

```python
# Complete image analysis pipeline
model.operation = "vqa"
model.prompt = "Describe this image comprehensively"
dataset.apply_model(model, label_field="descriptions")

model.operation = "detect" 
model.prompt = "Find all objects, people, and important elements"
dataset.apply_model(model, label_field="objects")

model.operation = "classify"
model.prompt = "Classify scene type, mood, lighting, and setting"
dataset.apply_model(model, label_field="scene_attributes")

model.operation = "ocr"
model.prompt = "Extract any visible text or signage"
dataset.apply_model(model, label_field="text_content")
```

## License

This integration is subject to the NVIDIA Open Model License Agreement and Llama 3.1 Community Model License.

# Citation

```bibtex
@misc{llama3.1-nemotron-nano-vl,
 title={Llama-3.1-Nemotron-Nano-VL-8B-V1},
 author={NVIDIA},
 year={2025},
 url={https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1},
}
```