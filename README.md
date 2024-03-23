# deep_learing_with_low_code_libraries

Files and the Content Inside them
1. FastAI_Vision2.ipynb
 - This notebook includes using fastai to train models and making inferences for the following tasks
  1. Classification - on Tiny MNIST dataset
  2. Image Segmentation - on Camvid Tiny data set
  3. Object Detection
4. FastAI3.ipynb
  - This notebook includes using fastai to train models and making inferences for the following tasks
  1. Learning on Tabular data -using synthetic data
  2. Building a recommendation model for song based on https://www.kaggle.com/datasets/rymnikski/dataset-for-collaborative-filters
3. huggingface_api.pynb
  - This notebook includes using huggingface API and transformers, for trying out transformers for various tasks such as :
  1. Text Classification - using text-classification pipeline 
  2. Named Entity Recognition - using ner pipeline
  3. Question Answering - using a question-answering pipeline
  4. Text Summarization - using the summarization pipeline
  5. Translation - using translation_en_to_fr pipeline and Helsinki-NLP/opus-mt-en-fr model. Translating from English to French and then the other way around
  6. Zero-shot classification - using zero shot classification pipeline and vicgalle/xlm-roberta-large-xnli-anli model. Asking the model identify if the text if about certain topics
  7. Computer Vision - object detection with facebook/detr-resnet-101-dc5 model
  8. Audio - audio to transcript conversion using facebook/wav2vec2-base-960h model
  9. Table QA - QA based on a given table using google/tapas-large-finetuned-wtq model
4. keras_nlp.ipynb
- This notebook includes using TensorFlow and KerasNLP for completing the following tasks
  1. Inference with a pre-trained classifier - bert_tiny_en_uncased_sst2
  2. Fine-tuning a pre-trained backbone - used threads app review data https://www.kaggle.com/datasets/saloni1712/threads-an-instagram-app-reviews. Fined tuned bert_tiny_en_uncased_sst2 backbone
  3. Fine-tuning with user-controlled preprocessing
  4. fine-tuning a custom model
5. keras+cv.ipnb
- This notebook includes using TensorFlow, KerasCV for the following tasks:
  1. Using YOLO for Object detection
  2. Inference Using a pre-trained classifier - used efficientnetv2_b0_imagenet_classifier
  3. Fine-tuning a pre-trained classifier - for classifying images of horses and humans. Used efficientnetv2_b0_imagenet
  4. Training on custom image classifier  - for classifying images of horses and humans. Used EfficientNetV2B0Backbone and the sequential layer API of KerasCV
  5. Training a Custom Object detection model using YOLOV8 backbone, and using the pascal_2007 dataset. 
