# Endoscopy Image Segmentation using U-Net
[![GitHub license](https://img.shields.io/github/license/jassatish/Endoscopic-Landmark-and-Polyp-Detection-Using-Kvasir-Dataset)](https://github.com/jassatish/Endoscopic-Landmark-and-Polyp-Detection-Using-Kvasir-Dataset/blob/main/LICENSE)

This project implements a U-Net model for segmenting medical images from endoscopic procedures. The model is trained to identify and isolate key regions within endoscopic images, which can assist in diagnosing conditions in the gastrointestinal tract.
## Dataset

The dataset used in this project is from the Kvasir-Seg dataset, which contains labeled images from gastrointestinal endoscopy procedures. To use this dataset, download it from [link to dataset source] and place it in the `data/` folder as specified in the notebook.
## Installation

First, clone this repository and navigate to the project directory:
```bash
git clone https://github.com/jassatish/Endoscopic-Landmark-and-Polyp-Detection-Using-Kvasir-Dataset.git
cd Endoscopic-Landmark-and-Polyp-Detection-Using-Kvasir-Dataset

pip install -r requirements.txt

```
## Usage

Open the Jupyter Notebook:
```bash
jupyter notebook endoscopy.ipynb
```
The notebook is organized as follows:

1. Data Preprocessing: Loads and preprocesses the images and masks.

2. Model Definition: Defines and compiles the U-Net model.

3. Training: Trains the model on the preprocessed data.

4. Evaluation: Evaluates the model and displays metrics like accuracy, loss, and AUC.

5. Predictions and Visualizations: Shows example predictions with Grad-CAM visualizations for model interpretation.
Results


## Results

- **Accuracy:** 94%
- **AUC (ROC curve):** 0.97
- **Example Predictions:** The model successfully segments key regions in endoscopic images. See the notebook for example outputs, including Grad-CAM visualizations.

## Links to Data
https://www.kaggle.com/datasets/abdallahwagih/kvasir-dataset-for-classification-and-segmentation/code
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
