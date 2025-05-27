# Speech Emotion Recognition: An Empirical Analysis of Machine Learning Algorithms Across Diverse Datasets
***


# 📌 Project Overview
Communication is expressing one’s feelings, ideas, and thoughts. Speech is a primary medium for communication. While people communicate with each other in several human interactive applications, such as a call center, entertainment, E-learning between teachers and students, medicine, and communication between clinicians and patients (especially important in the field of psychiatry), it is crucial to identify people’s emotions to better understand what they are feeling and how they might react in a range of situations. Automated systems are constructed to recognise emotions from analysis of speech or human voice using Artificial Intelligence (AI) or Machine Learning (ML) approaches, and these approaches are gaining momentum in recent research.

This project presents an empirical analysis of machine learning approaches, specifically Support Vector Machine (SVM), to accurately classify human emotions from speech signals. We extract acoustic features and train the system on three popular datasets: **RAVDESS**, **TESS**, and **SAVEE**.

With this, we aim to push the boundaries of automated emotion recognition in healthcare, education, entertainment, and human-computer interaction.


# 💡 Motivation
Emotions are vital in human communication, influencing decision-making and interpersonal interactions. Speech is one of the most natural and expressive media, so recognizing emotions from voice has gained increasing attention in machine learning.

Speech Emotion Recognition (SER) enables machines to detect emotional cues from speech signals, with applications in:

    🎓 E-learning: Identifying students' emotions to adapt teaching methods and improve engagement.

    📞 Call Centers: Analyzing conversations to monitor customer satisfaction and agent performance.

    🏥 Telemedicine: Understanding patients’ emotional states during remote consultations, especially in mental health support.

# 🎯 Aim & Objectives
## 🔍 Aim
To develop a machine learning-based Speech Emotion Recognition (SER) system that accurately identifies human emotions from speech signals using advanced audio feature extraction and classification techniques.

## ✅ Objectives
#### 🗣️ Recognize key emotions such as: Happy, Sad, Angry, Fear, Disgust, Surprise, Calm, and Neutral from speech.

#### 🎧 Extract meaningful audio features, including:

* Mel Frequency Cepstral Coefficients (MFCC)
* Chroma Features
* Mel-Spectrogram
* Spectral Centroid, Bandwidth, Roll-off
* Zero Crossing Rate (ZCR)
* Root Mean Squared Energy (RMSE)

#### 🤖 Train and evaluate a Support Vector Machine (SVM) classifier for emotion prediction.

#### 📊 Test the model on benchmark datasets:

* RAVDESS
* TESS
* SAVEE

#### 🧪 Create and analyze a mixed dataset from the above to improve generalization and performance.

#### 🚀 Deploy a scalable solution suitable for educational, clinical, and customer service applications.

# 🛠️ Proposed Methodology
In this work, we propose a system that automatically identifies humans' emotional states from their speech signals based on the embellishment of the signal. The whole process is divided into the following parts: 
* **Dataset Preparation**, 
* **Feature Extraction**, 
* **Classification**, and 
* **Emotion Recognition**.

<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/93c5ec02-ae52-4f16-9593-ad792e58759d" width="800"/></td>
  </tr>
</table>
<p align="center">Fig.1. Generalized block diagram of Speech Emotion Recognition system.</p>


### 📁 1. Dataset
We used three benchmark emotional speech datasets along with a combined dataset to improve performance and generalization:

* **RAVDESS**: 1440 speech samples from 24 actors, covering 8 emotions (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprise).

* **TESS**: 2800 audio samples from two female speakers, with 7 emotions (excluding Calm).

* **SAVEE**: 480 samples from 4 male British speakers, covering 7 emotions.

* **Combined Dataset**: Merged version of the above datasets totaling 4720 audio files across 8 emotions, used to train a more robust model.


### 🎧 2. Feature Extraction
Raw speech signals are converted into meaningful numerical representations using Librosa, a Python audio analysis library. We extracted the following features:

|     Features     |     Short Description     |
|:-----------------------:|:------------------------:|
|        MFCC    |        Represents the short-term power spectrum of sound.       |
|        Chroma        |        Captures the 12 pitch classes of the audio signal.        |
|       Mel-Spectrogram | Displays how spectral energy is distributed over the Mel scale.        |
|       Spectral Centroid | Indicates the spectrum's brightness or center of mass.     |
|       Spectral Bandwidth | Measures the spread of the spectrum around the centroid.      |
|        Spectral Roll-off | Frequency below which 85% of the total energy lies.      |
|       RMSE | Represents energy by computing the root mean square of the signal amplitude.     |
|     Zero Crossing Rate | Counts how often the signal changes sign (crosses zero).    |

All these features are concatenated into a feature vector for classification.

### 🤖 3. Classification
We used a Support Vector Machine (SVM) classifier to categorize speech samples into one of the emotional classes. SVM is chosen because:

* It handles high-dimensional data efficiently.

* It is effective in cases with clear class boundaries.

* It is memory-efficient, using only a subset of training points (support vectors).

* It supports different kernel functions for better decision boundaries.

### 🧠 4. Emotion Recognition
The trained SVM model takes in feature vectors derived from speech signals and classifies them into emotional categories. The model was trained and tested on individual and combined datasets, achieving high accuracy across all evaluations.

# 📊 Results and Discussion
### 🔬 4.1 Experiment Setup
All implementations were carried out in Python. Experiments were conducted on three standard speech emotion datasets—RAVDESS, TESS, and SAVEE—and on a combined dataset formed by merging all three to improve generalizability.

### 🔧 Preprocessing Steps:
* Pre-emphasis Filtering: A filter with coefficient α = 0.97 was applied to amplify high-frequency components.

* Framing and Windowing:

    1. Speech signals were divided into overlapping frames.

    2. A Hamming window was applied to each frame to minimize spectral leakage and discontinuities.

### 🎛 Feature Engineering:
Eight acoustic features were extracted from the filtered and windowed signals:

* MFCC, Chroma, Mel-Spectrogram, Spectral Centroid, Bandwidth, Roll-off, RMSE, and ZCR.

### 🧪 Model Training:
* Dataset split: 80% training, 20% testing

* Classifier used: Support Vector Machine (SVM)

* Evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC**, and **AUC**.


### 📈 4.2 Results and Analysis
#### 🧠 Performance by Dataset:

Dataset | Accuracy (%) | Precision (%) | Recall (Sensitivity) (%) | F1-Score (%)
-- | -- | -- | -- | --
RAVDESS | 99.59 | ~99 | 98.50 | ~99
TESS | 99.82 | ~100 | 99.57 | ~100
SAVEE | 98.95 | ~98 | 97.43 | ~98
Combined | 100.00 | 100.00 | 100.00 | 100.00

* The combined dataset showed the best performance across all metrics.

* The TESS dataset outperformed others in individual evaluations due to larger and balanced samples.

* High AUC (Area Under Curve) score: 99% macro average, reaching 100% for the combined dataset, reflecting excellent classifier performance.

#### 📊 Visual Analysis:

**Comparative metrics (Accuracy, Precision, Recall, F1-score) for all datasets.**
<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/7919eaad-b41b-4188-881e-5cf437b6862d" width="600"/></td>
  </tr>
</table>
<p align="center">Fig.2. Comparison among four datasets based on the accuracy, precision, recall, and F1-score metrics.</p>


**Confusion matrices highlight per-class classification performance.**
<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/890dfa56-a7da-4dc6-9769-7943d8acdace" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/e834bb37-b39a-44ec-bead-c9f2a4a7b812" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c474e77f-b62e-4641-b425-97e12acc2900" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/649fad89-9004-4b9f-b082-eb280e2c4ca1" width="400"/></td>
  </tr>
</table>
<p align="center">Fig.3. Confusion Matrix of the model for the three different datasets and their combinations.</p>


**ROC curves for each class in all datasets confirm excellent generalization and separability.**
<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/4a950913-fbed-4404-b371-ecddffd630b8" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/57e4bf91-f8bb-450d-9681-5e2a7678e7fd" width="400"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/8d1eb62e-e15d-47a1-878b-b5ace70ba973" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/3e85e2d3-9338-417f-863c-1c4ccfac699d" width="400"/></td>
  </tr>
</table>
<p align="center">Fig.4. ROC performance curves of the proposed emotion classification model in all four datasets.</p>


### 🆚 4.3 Comparison with State-of-the-Art Methods
The proposed SVM-based model was benchmarked against existing speech emotion recognition systems.
Dataset | Accuracy of Proposed Model | Accuracy of Prior Methods
-- | -- | --
RAVDESS | 99.59% | < 98%
TESS | 99.82% | ~98%
SAVEE | 98.95% | ~96%


<table align="center">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eef2be93-947e-475c-bdd0-d45d78a4e277" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/e394bfb4-b945-4217-b612-a901f26420eb" width="400"/></td>
    <td><img src="https://github.com/user-attachments/assets/3ae921f6-a5de-431b-8eae-378bc04b290e" width="400"/></td>
  </tr>
</table>
<p align="center"> Fig.5. Performance Evaluation of the Proposed Model Compared with Existing Methods.</p>



* Significant improvements in all datasets.
* The proposed method is robust and scalable, achieving state-of-the-art results, especially on larger datasets.


## 🛠️ Technologies Used
* Python 3.x
* Librosa for feature extraction
* Scikit-learn for model building
* Matplotlib/Seaborn for visualization

## 📚 Citation
If you use this work in your research, please cite:
Ahammed, M. et al. (2024). Speech Emotion Recognition: An Empirical Analysis of Machine Learning Algorithms Across Diverse Data Sets. In: Mahmud, M., Ben-Abdallah, H., Kaiser, M.S., Ahmed, M.R., Zhong, N. (eds) Applied Intelligence and Informatics. AII 2023. Communications in Computer and Information Science, vol 2065. Springer, Cham. https://doi.org/10.1007/978-3-031-68639-9_3.

# 📬 Contact
### 📧 mostafizs154@gmail.com
### 🔗 [LinkedIn](https://www.linkedin.com/in/mostafiz-ahammed-7b4266106/)
