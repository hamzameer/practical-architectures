# Practical Architectures for Deep Learning - Day 2 PM
**Vision Transformers and their Applications**


This repository contains the exercises for our session on the Transformer architecture, Vision Transformers and their applications in video analysis.

## 0. Set-up
### 0.1 Load environment
```bash
# 1. start from home
cd ~

# 2. clone exercise files
git clone https://github.com/hamzameer/practical-architectures.git

# 3. change into exercise file directory
cd practical-architectures

# 4. load python
module load python

# 5. activate conda
source activate

# 6. activate the exercise environment
conda activate /project/nanocourse/PracArchDL/shared/Day2Noon/env

# 7. set environment variables
source set_envs.sh

# 8. run environment health check script
python ./check.py
```

**Raise your hand if there are are any error messages after running `check.py`**

### 0.2 Start Jupyter Lab

Assuming everything worked with the environment check let's open up jupyter lab and navigate to the exercise directory

First, start jupyter lab with the following command:
```bash
jupyter lab
```

If there were no errors, then a browser window should open.  In the lefthand side, open the folder called `notebooks`.

Once open, you sould see four numbered exercise notebooks.  Please open the first notebook, labeled `00-check-jupyter.ipynb` and run the first cell.  If you run into any errors executing this cell, raise your hand.  If you've gotten this far, error-free then we are ready to get started with this afternoon's exercises.

## Exercise 1: Review of Transformer Architecture using minGPT

### 1.1 Background: History of Generative Pre-trained Transformers (GPTs)

The advent of Generative Pre-trained Transformers (GPTs) represents a significant milestone in the development of language models. Originating from OpenAI, the GPT series has evolved through several iterations, each building upon the last to enhance the model's understanding and generation of human language. A brief summary of their iterations since their original publication in 2018 is below for backgound.

#### 1.1.1 GPT-1: The Inception
The journey of GPTs began with the introduction of the first Generative Pre-trained Transformer (GPT-1) in June 2018. As an initial foray into transformer-based models, GPT-1 was one of several models (ULMFiT, BERT) released around this time participating in a paradigm shift for training language models using unsupervised learning before fine-tuning them on specific tasks.  While transfer learning was very common with CNNs in the vision domain, transfer learning was much less commonly used in NLP tasks, especially with a unsupervised pretraining step.

- Foundational Paper: ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

#### 1.1.2 GPT-2: Advancements and Controversies
With its release in February 2019, GPT-2 marked a substantial advancement in the field, expanding the model to 1.5 billion parameters. This increase enabled the model to generate more coherent and contextually relevant text. However, due to potential risks associated with its capabilities, OpenAI made the controversial decision to initially limit the release of the full model.

This decision sparked considerable debate within the AI community about the ethics of AI development and the balance between innovation and responsible disclosure. Critics argued that the move was more of a publicity stunt than a genuine safety measure, while proponents lauded the caution exercised by OpenAI in the face of potential societal impacts.

- Comprehensive Study: ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

#### 1.1.3 GPT-3: Scaling to New Heights
The introduction of GPT-3 in June 2020 represented a monumental leap in the series, scaling up to a previously unheard of 175 billion parameters. This escalation in scale allowed GPT-3 to exhibit a broad range of capabilities, often requiring minimal task-specific data to perform tasks at near-human levels.

- In-depth Analysis: ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165)

#### 1.1.3 GPT-4: The Current Frontier
While specific details and papers may not be publicly available, GPT-4 most likely surpasses its predecessors in size and studies show significant increases in capability, and adaptability.

- [OpenAI Blog: GPT-4 Research](https://openai.com/research/gpt-4): Refer to OpenAI's official blog for the latest information, including any new publications or technical reports on GPT-4.

### 1.2 Objectives

We'll take a deep dive into the intricacies of the GPT architecture through a detailed examination of Andrej Karpathy's MinGPT `model.py` file. This is designed to provide you with a practical and thorough understanding of the mechanics behind Generative Pre-trained Transformers. MinGPT, a minimalistic yet fully-functional implementation of the GPT architecture, strips down the complexity to its essential components, making it an ideal educational tool. 

By dissecting and analyzing the `model.py`` file, we will uncover the nuances of transformer-based models, delve into the subtleties of self-attention mechanisms, and understand how these components coalesce to create a powerful language model. Whether you're a seasoned AI practitioner or a student stepping into the world of machine learning, this walkthrough will enhance your comprehension of one of the most influential architectures in modern NLP. [github repo here](https://github.com/karpathy/minGPT)

## Exercise 2: Train a small minGPT Q&A model

### 2.1 Overview: Training a Medical Q&A Model with MinGPT
In this exercise, we will utilize the MinGPT library to train a basic medical Question and Answer (Q&A) model from scratch. Our goal is to create a model that can provide answers to medical-related questions using the MedQA dataset. While our resulting model may not reach the precision required for clinical use, this exercise will serve as a valuable hands-on experience in understanding the training process of Q&A models and the nuances of handling domain-specific data.

**DISCLAIMER:** We will be training this small, toy model from scratch(!) on a relatively small dataset (9K examples) so DO NOT use this model for any diagnostic purposes!  This exercise is only intended to demonstrate how to use PyTorch directly for training a model from scratch.


### 2.2 Open-Domain Machine Q&A Models
Open-domain machine Q&A models are designed to answer questions on a wide range of topics without being restricted to a specific domain or set of documents. These models utilize large corpora of text to learn how to understand and generate responses to queries. They are often pre-trained on a broad dataset and can be fine-tuned for more specialized domains, such as medical information. Their versatility makes them powerful tools in the field of information retrieval and natural language understanding.

For more background, please refer to these references:

1. Chen, D., Fisch, A., Weston, J., & Bordes, A. (2017). 
Reading Wikipedia to Answer Open-Domain Questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL). [Link to Paper](https://arxiv.org/abs/1704.00051)

2. Roberts, A., Raffel, C., & Shazeer, N. (2020). How Much Knowledge Can You Pack Into the Parameters of a Language Model? (T5 model paper which discusses open-domain Q&A) [Link to Paper](https://arxiv.org/abs/2002.08910)
 
 ### 2.3 The MedQA Dataset
The MedQA dataset, introduced in the paper by Jin et al. in 2021, is a comprehensive dataset formulated for the task of medical question answering. It consists of question-and-answer pairs derived from a variety of medical examinations and literature, providing a robust foundation for training models intended for medical information retrieval and Q&A tasks. The dataset encompasses a broad spectrum of medical knowledge, making it an excellent resource for training specialized models in the medical domain. Its structured format facilitates the training of models capable of interpreting medical queries and providing accurate, informative responses.

See Jin (2021) for more info:

1. Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2021). Disease Knowledge Distillation for Medical Dialogue Generation.  [Link to Paper](https://arxiv.org/abs/2109.00704)

## Exercise 3: Generate Dino V2 Embeddings of Sample Video

### 3.1 Overview: Unsupervised Action Segmentation
In this final exercise, we turn our attention to using Transformers for video analysis. Our task involves performing unsupervised action segmentation on a video that captures a simulated doctor-patient interaction during a patient exam for Deep Vein Thrombosis (DVT). The goal is to identify distinct phases or actions within the interaction, leveraging the capabilities of a Vision Transformer (ViT) and clustering algorithms.

The sample video that we will analyze can be found here on YouTube: [Step 2 CS Patient Encounter, DVT](https://youtu.be/6Hoyk73SCrg?si=gPEyt7b0lQwAnB9_)

Unsupervised action segmentation is a crucial task in video analysis, aiming to divide a video into segments based on different actions or events occurring in the sequence. It holds significant value in various fields, including medical training and assessment, where it can help categorize and evaluate specific procedural elements. In our case, segmenting a doctor-patient interaction during a DVT examination can aid in educational analysis and feedback for medical practitioners.

### 3.2 Pre-trained Vision Transformer (ViT)
For this task, we'll employ a specific Vision Transformer known as DinoV2. Vision Transformers have revolutionized image recognition tasks by applying the transformer architecture, originally designed for NLP, to visual inputs. A pre-trained ViT like DinoV2 is adept at encoding each frame of a video into a high-dimensional vector representation. These embeddings capture rich, contextual visual information, enabling us to understand and analyze the content of each frame at a granular level.

More information about the DinoV2 model from Meta AI can be found here: [Very cool paper page for DinoV2](https://dinov2.metademolab.com) 

We will be relying on the phenomenal Huggingface Transformers library to provide the model code and weights.  You can read more about this library here: [Huggingface Transformers](https://huggingface.co/docs/transformers/index).  If you intend to do a lot of work with Transformers, getting to know this library well is an absolute must.

### 3.3 K-means Clustering for Segment Identification
Once we have our frame embeddings, the challenge lies in identifying semantically meaningful segments without prior knowledge or labels. Here, k-means clustering comes into play. This unsupervised learning algorithm groups data points (in our case, frame embeddings) into k clusters based on their similarity. By applying k-means clustering to the embeddings, we can discover patterns and group frames into clusters that represent distinct actions or phases in the video. This method allows us to automatically identify and segment the video into parts corresponding to different aspects of the patient examination.

The specific library that we will use for this clustering can be found here: [Torch Spatial K-means](https://github.com/mike-holcomb/torch-spatial-kmeans)

Here is a survey paper reviewing other potential methods and architectures that attempt to tackle this task: [Temporal Action Segmentation: An Analysis of Modern Techniques](https://arxiv.org/abs/2210.10352)

