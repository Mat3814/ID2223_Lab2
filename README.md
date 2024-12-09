# Lab 2 ID2223 - FineTuned LLMs

This repo is part of the KTH course ID2223, composed of two tasks

## Task 1 : Fine-tune a model for language transcription, add a UI

### Plan :
The work is divided in 3 notebooks

### ID2223_Lab2_Finetune_Model.ipynb

This notebook fine-tunes the pretrained unsloth/Llama-3.2-3B-Instruct model using LoRA (Low-Rank Adaptation) for efficient parameter tuning on the FineTome-100k dataset. The fine-tuned model, optimized for memory usage with 4-bit quantization, is trained using supervised fine-tuning (SFT) and then saved and uploaded to Hugging Face for future use.

### ID2223_Lab_2_Restart_Training.ipynb

The code reuses a previous adapter by loading it from a checkpoint (llama_lora_adapter) and resumes the training process from that point. This allows for continuing optimization without starting from scratch

### ID2223_Lab2_Merge_and_Infer.ipynb

The script merges a base LLaMA model with a LoRA adapter to fine-tune it, converts the merged model into GGUF format for efficient CPU execution, and uploads it to the Hugging Face Hub. This enables deployment of a quantized and fine-tuned model optimized for low-resource environments.

## Usage

You can use the adapter through this link : [ID2223-Lab/llama_lora_adapter](https://huggingface.co/ID2223-Lab/llama_lora_adapter)

You can use the model from this link : [ID2223-Lab/llama_lora_merged_GGUF](https://huggingface.co/ID2223-Lab/llama_lora_merged_GGUF)

You can interact with the chat bot on the hugging face space : [ID2223-Lab/iris](https://huggingface.co/spaces/ID2223-Lab/iris)

## Task 2 : Improve pipeline scalability and model performance

## Subject

The goal of this task was to finetune with scalability and performances, a model for a precise task.

### Plan :
The work is divided in 2 notebooks


### ID2223_Lab2_Translation_Model_creation.ipynb

This script fine-tunes a google/t5small model on the opus_books dataset for English-to-French translation using the Hugging Face transformers library. It prepares the data, trains the model with specific training arguments, and evaluates its performance, before using the trained model to translate a sample text from English to French.

### ID2223_Lab2_Translation_Model_upgrade.ipynb

This script restart fine-tunes a T5 model for English-to-French translation using the Opus Books dataset, resuming from a previous checkpoint. It prepares the data, tokenizes it, and continues training the model while evaluating its performances.

## Improvement Keys

**Model Architecture**

We used **T5 Small**, a compact version of the T5 model, which is ideal for translation tasks as it treats them as text-to-text problems. Its smaller size enables faster inference on CPUs while maintaining good translation quality, making it efficient for resource-limited environments.

Model hyperparameters : 

**learning_rate**: 2e-5 - The learning rate for the optimizer, which controls the step size during gradient descent.
**batch_size** : 16 - the number of training examples utilized in one forward and backward pass during training
**weight_decay**: 0.01 - The rate of weight decay applied for regularization to prevent overfitting.
**fp16**: True - Indicates whether 16-bit floating-point precision will be used to speed up training and reduce memory usage.

**Data Approach**

We began with the **Opus Book** dataset and expanded to the **Europarl** dataset for English-French translation. The combination of these datasets provides a diverse range of sentence structures and vocabulary, improving the model's ability to generalize and perform better in real-world translation tasks.

## Results

BLEU score : The BLEU score is a metric used to evaluate the quality of machine-generated translations 

![alt text](https://github.com/Mat3814/ID2223_Lab2/blob/main/Task2/Metrics/Bleu_curve.png)

From 24 we started switching from opus book to europarl, here we saw clear improvements on the score

## Usage

You can use the model on this link : [Mat17892/t5small_enfr_opus](https://huggingface.co/Mat17892/t5small_enfr_opus)

You can interact with the translator on the hugging face space : [Heit39/UI_TranslationLLM](https://huggingface.co/spaces/Heit39/UI_TranslationLLM)