# Vision Face Matcher Model

This repository provides two equivalent scripts to convert the VGG-Face model into a truncated version saved in the ONNX format. This conversion is useful for optimizing face representation tasks (verification). You can choose to run the script either as a Python file for local execution or as a Jupyter Notebook for interactive use.

## Table of Contents

- [Vision Face Matcher Model](#vision-face-matcher-model)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Architecture](#model-architecture)
  - [Files](#files)
  - [References](#references)
  - [License](#license)

## Installation

To run the scripts, you need to have the following installed:

- Python (3.6 or later)
- TensorFlow (version 2.x preferred)
- tf2onnx (for converting the model to ONNX format)

You can install the required Python packages with the following command:

```bash
pip install -U tensorflow tf2onnx
```

## Usage

1. **Download the VGG-Face Weights:**

    Download the pre-trained VGG-Face model weights using the command:

    ```bash
    wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
    ```

2. **Run the Script:**

    There are two options to run the script, providing flexibility based on your environment:

    - **Python Script:**
        Execute the `model_generator_vgg_face_to_onnx_truncated.py` file in your local Python environment:

        ```bash
        python model_generator_vgg_face_to_onnx_truncated.py
        ```

    - **Jupyter Notebook:**
        Open and run the `Model_Generator_VGG_Face_to_ONNX_TRUNCATED.ipynb` file using Jupyter Notebook for an interactive experience.

3. **Convert Model to ONNX:**

    To convert the model to the ONNX format, use the following command (assumed to be handled within the scripts):

    ```bash
    python -m tf2onnx.convert --saved-model saved_vgg_face_model_truncated --output saved_vgg_face_model_truncated.onnx --opset 13
    ```

4. **Saving the Model:**

    After conversion, you will have the generated `saved_vgg_face_model_truncated.onnx` file.

## Model Architecture

The VGG-Face model architecture is defined as a Sequential model in TensorFlow. The model is originally designed to classify 2622 identities. The conversion process truncates the model to output embeddings instead of classifications.

- **Convolutional Layers**: The model includes several convolutional layers with `ReLU` activations.
- **Pooling Layers**: Max-pooling layers reduce spatial dimensions.
- **Dense Layers**: Fully connected layers are followed by dropout for regularization.

The truncated model outputs embeddings from the second-to-last layer, bypassing the final softmax layer used for classification.

## Files

- `Model_Generator_VGG_Face_to_ONNX_TRUNCATED.ipynb`: Jupyter Notebook for interactive execution.
- `model_generator_vgg_face_to_onnx_truncated.py`: Python script for local execution.

## References

- VGG-Face: [GitHub Repository](https://github.com/serengil/deepface)
- TensorFlow: [Official Site](https://www.tensorflow.org/)
- tf2onnx: [GitHub Repository](https://github.com/onnx/tensorflow-onnx)

## License

This project is open-source and available under the MIT License.

For more information, refer to the original implementation by 2060.io and their open-source tools for building verifiable credential-based services.
