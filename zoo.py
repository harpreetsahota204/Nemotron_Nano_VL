
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import ast 

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "answer": # The answer,
            "bbox": (x1, y1, x2, y2), # 'bbox' refers to the bounding box position of the 'answer' content in the image. bbox is the coordinates of the top-left corner and the bottom-right corners. The 'bbox' should be normalized coordinates ranging from O to 1000 by the image width and height.
            "reason": # The reason for your answer
        },
    ]
}
```

- Provide specific answer for each detected element but limit to one or two words
- Include all relevant elements that match the user's request
- For UI elements, include their function when possible (e.g., "Login Button" rather than just "Button")
- If many similar elements exist, prioritize the most prominent or relevant ones

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and detect.
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label",
            "reason": # The reason for your answer
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image, but limited to one or two word responses
- The response should be a list of classifications
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class NemotronNanoModel(SamplesMixin, Model):
    """A FiftyOne model for running Nemotron Nano VLM vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }
        
        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading processor")
        self.processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            device=self.device,
            use_fast=True
        )

        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit - clear GPU memory cache."""
        # Clear cache based on device type (don't move model to CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False
    

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        This method handles JSON within markdown code blocks (```json) or raw JSON strings.
        
        Args:
            s: Raw string output from the model containing JSON
                
        Returns:
            Dict: The parsed JSON content 
            None: If parsing fails
            Original input: If input is not a string
        """
        # Return non-string inputs as-is
        if not isinstance(s, str):
            return s
        
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            try:
                # Split on markdown markers and take JSON content
                json_str = s.split("```json")[1].split("```")[0].strip()
            except:
                json_str = s
        else:
            json_str = s
            
        # Attempt to parse the JSON string
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except:
            # Log parsing failures for debugging
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return None

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections with associated reasoning.
        
        Takes detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization from 0-1000 range to 0-1 range
        - Label extraction
        - Reasoning attachment
        
        Args:
            boxes: Detection results, either:
                - List of detection dictionaries
                - Dictionary containing 'data' and 'reasoning'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
        """
        detections = []
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("detections", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure we're working with a list of boxes
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each bounding box
        for box in boxes:
            try:
                # Extract bbox coordinates, checking both possible keys
                bbox = box.get('bbox', box.get('bbox_2d', None))
                if not bbox:
                    continue
                
                # Extract reasoning for this specific detection
                reasoning = box.get("reason", "")
                
                # Handle bbox coordinates that are passed as strings like "(100,260,940,690)"
                if isinstance(bbox, str):
                    bbox = ast.literal_eval(bbox)
                    
                # Convert coordinates to float
                x1_norm, y1_norm, x2_norm, y2_norm = map(float, bbox)
                
                # Directly convert from 0-1000 range to 0-1 range for FiftyOne
                x = x1_norm / 1000.0  # Left coordinate (0-1)
                y = y1_norm / 1000.0  # Top coordinate (0-1)
                w = (x2_norm - x1_norm) / 1000.0  # Width (0-1)
                h = (y2_norm - y1_norm) / 1000.0  # Height (0-1)
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=str(box.get("answer", box.get("label", "object"))),  # Handle both answer and label keys
                    bounding_box=[x, y, w, h],
                    reasoning=reasoning  # Attach reasoning to detection
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications with reasoning.
        
        Processes classification labels and associated reasoning into FiftyOne's format.
        
        Args:
            classes: Classification results, either:
                - List of classification dictionaries
                - Dictionary containing 'data' and 'reasoning'
                
        Returns:
            fo.Classifications: FiftyOne Classifications object containing all results
            
        Example input:
            {
                "data": [{"label": "cat"}, {"label": "animal"}],
                "reasoning": "Image shows a domestic cat"
            }
        """
        classifications = []
        
        # Handle nested dictionary structures
        if isinstance(classes, dict):
            classes = classes.get("classifications", classes)
            if isinstance(classes, dict):
                classes = next((v for v in classes.values() if isinstance(v, list)), classes)
        
        # Process each classification
        for cls in classes:
            try:
                # Extract reasoning for this specific classification
                reasoning = cls.get("reason", "")
                
                # Create FiftyOne Classification object
                classification = fo.Classification(
                    label=str(cls["label"]),
                    reasoning=reasoning  # Attach reasoning to classification
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        input_width, input_height = image.size
        
        image_features = self.processor([image])

        generation_config = dict(
            max_new_tokens=4096, 
            do_sample=False, 
            pad_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.model.chat(
            tokenizer=self.tokenizer, 
            question=prompt, 
            system_prompt = self.system_prompt,
            generation_config=generation_config,
            **image_features)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "detect":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)
