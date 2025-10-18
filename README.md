# Dermatological Disease Classification using TensorFlow

This project is a deep learning solution for the classification of dermatological diseases from images. It utilizes a convolutional neural network (CNN) built with TensorFlow, leveraging the state-of-the-art EfficientNet architecture through transfer learning.

The model is trained to accurately identify 21 distinct skin conditions, demonstrating a practical application of computer vision in providing preliminary diagnostic insights.

## Key Features
- **Comprehensive Classification**: Identifies 21 distinct dermatological conditions.
- **High Accuracy**: Utilizes the state-of-the-art EfficientNet model for robust performance.
- **Confidence Scoring**: Provides a confidence score for each prediction.
- **Simple Interface**: Easy to use via a command-line interface.

## Model Performance
The model was trained and evaluated on a diverse dataset of dermatological images.

* **Validation Accuracy:** `75%`

#### Classifiable Conditions
The model can identify the following 21 skin conditions:

| | | |
| :--- | :--- | :--- |
| Acne | Actinic Keratosis | Benign Tumors |
| Bullous | Candidiasis | Drug Eruption |
| Eczema | Infestation & Bites | Lichen Planus |
| Lupus | Moles | Psoriasis |
| Rosacea | Seborrheic Keratoses| Skin Cancer |
| Sun Damage | Tinea | Vascular Tumors |
| Vasculitis | Vitiligo | Warts |

## Technologies Used
- `Python`
- `TensorFlow` & `Keras`
- `scikit-learn`
- `NumPy` & `Pandas`
- `Matplotlib` for visualization

## Prediction Report
[!Classes Report](/class_prediction_report.png)

## Setup and Installation
To run this project locally, follow these steps:

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To predict a single image, run the `predict.py` script from the root directory:
```sh
python predict.py --image /path/to/your/image.jpg