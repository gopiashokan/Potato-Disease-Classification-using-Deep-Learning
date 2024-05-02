# Potato Disease Classification using Deep Learning

### Introduction

In the agriculture industry, farmers often face challenges in identifying diseases in potato plants, such as early blight, late blight, or determining if the plant is healthy. This uncertainty makes it difficult for farmers to apply the appropriate fertilizers and treatments, impacting crop yield and quality. To address this issue, we have developed a deep learning model using TensorFlow to classify images of potato plants, aiding in the accurate identification of diseases. By leveraging machine learning technology, our solution aims to improve agricultural practices, optimize resource allocation, and ultimately enhance the production of healthy potato plants.

<br />

### Table of Contents

1. Key Technologies and Skills
2. Installation
3. Usage
4. Features
5. Contributing
6. License
7. Contact

<br />

### Key Technologies and Skills
- Python
- TensorFlow
- Keras
- Convolutional Neural Network (CNN)
- Numpy
- Matplotlib
- Pillow
- Streamlit

<br />

### Installation

To run this project, you need to install the following packages:

```python
pip install numpy
pip install tensorflow
pip install matplotlib
pip install pillow
pip install streamlit
pip install streamlit_extras
```

**Note:** If you face "ImportError: DLL load failed" error while installing TensorFlow,
```python
pip uninstall tensorflow
pip install tensorflow==2.12.0 --upgrade
```

<br />

### Usage

To use this project, follow these steps:

1. Clone the repository: ```git clone https://github.com/gopiashokan/Potato-Disease-Classification-using-Deep-Learning.git```
2. Install the required packages: ```pip install -r requirements.txt```
3. Run the Streamlit app: ```streamlit run app.py```
4. Access the app in your browser at ```http://localhost:8501```

<br />

### Features

#### Data Collection:
   - We obtained the potato disease image dataset from Kaggle, a renowned platform for datasets and data science resources. This dataset consists of images depicting diseased potato plant leaves, meticulously labeled into categories such as early blight, healthy, and late blight.

   - This collection serves as a valuable asset for training and evaluating our deep learning model, facilitating the development of an effective solution for potato disease classification.

📙 Dataset Link: [https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)


#### Preprocessing:

   - **Image Reading and Resizing:** We initiate the preprocessing phase by leveraging TensorFlow to read all images from the directory. Each image undergoes resizing to a standardized dimension of 256x256 pixels. Furthermore, we organize the processed images into batches with a size of 32, thus forming a structured dataset ready for subsequent analysis.

   - **Dataset Splitting:** To facilitate comprehensive model evaluation, we partition the dataset into three distinct subsets: training, validation, and testing. This segmentation ensures the robustness of our model's performance assessment by enabling separate training, validation, and testing phases, thus minimizing the risk of overfitting and enhancing generalization capabilities.

   - **Data Pipeline Optimization:** In pursuit of efficient model training, we optimize the data pipeline using TensorFlow's built-in functionalities. The `cache` function is strategically employed to circumvent the repetitive loading and reading of training images across epochs. Concurrently, the `prefetch` function enhances training speed by proactively preparing subsequent batches of training images. These optimizations collectively streamline the training process, resulting in significant time savings and improved computational efficiency.


#### Model Building and Training:

   - **Model Building:** We construct the model architecture using Keras, incorporating layers for resizing, rescaling, random flip, and random rotation to preprocess the input images. Additionally, a Convolutional Neural Network (CNN) architecture is implemented, comprising convolutional layers, pooling layers, and dense layers with adjustable filters/units and activation functions.

   - **Training:** During model training, we utilize the `Adam` optimizer, `sparse_categorical_crossentropy` loss function, and `Accuracy` metrics to optimize and evaluate the model's performance. The training process involves evaluating the model's performance on the validation dataset after each epoch, culminating in a final evaluation on the testing dataset. Upon completion of training, the model achieves an impressive accuracy of **97.8%**, signifying its capability to accurately classify potato disease images.


#### Model Deployment and Inference:
   - Following the completion of model training and evaluation, the trained model is saved to enable seamless deployment and inference on new images for classification purposes. To facilitate this process, a user-friendly Streamlit application is developed and deployed on the Hugging Face platform. 
   - This application empowers users to upload new images and obtain real-time classification results, providing a convenient interface for leveraging the model's capabilities in practical scenarios.

🚀 Application: [https://huggingface.co/spaces/gopiashokan/Potato-Disease-Classification](https://huggingface.co/spaces/gopiashokan/Potato-Disease-Classification)


![](https://github.com/gopiashokan/Potato-Disease-Classification-using-Deep-Learning/blob/main/image/Inference_image_output.JPG)


<br />

**Contributing**

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

<br />

**License**

This project is licensed under the MIT License. Please review the LICENSE file for more details.

<br />

**Contact**

📧 Email: gopiashokankiot@gmail.com 

🌐 LinkedIn: [linkedin.com/in/gopiashokan](https://www.linkedin.com/in/gopiashokan)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

