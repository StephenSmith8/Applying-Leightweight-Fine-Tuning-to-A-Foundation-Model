<h1> Applying-Lightweight-Fine-Tuning-to-A-Foundation-Model</h1>

<h2>Description</h2>
<b>The objective of this project was to bring together all of the essential components of a PyTorch+ Hugging Face training and inference process to accomplish the following:
<br />
<br />
  
- Load a pre-trained model and evaluate its performance
- Perform parameter-efficient fine tuning using the pre-trained model
- Perform inference using the fine-tuned model and compare its performance to the original model 
</b>

<h2>Languages and Utilities Used</h2>

- <b>Python</b>
- <b>Hugging Face PEFT Library</b>
- <b>PyTorch+</b>
- <b>LORA PEFT Technique</b>
- <b>GPT-2 Pre-trained Model</b>
- <b>Hugging Face Trainer</b>
- <b>Hugging Face Dataset</b>
 

<h2>Environments Used </h2>

- <b>Jupyter Notebook</b>

<h2>Project walk-through:</h2>
<h3>PHASE 1: LOADING AND EVALUATING A FOUNDATION MODEL</h3>
<b>The code depicted below primarily involves working with a transformer model for text classification using the Hugging Face transformers library. It performs tokenization, dataset loading, model setup, and training. Below, I will dive into and explain the function of each element of the program.
</b>
<br />
<br />
<img src="https://imgur.com/d01MfPY.png">


<h3>Importing Libraries</h3> 
<b>The code begins by importing libraries that contain the necessary elements for the code to function.
</b>
<br />
<br />
<img src="https://imgur.com/Eind0xy.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<ul>
  <li>GPT2Tokenizer: A tokenizer for GPT-2, used to convert text into tokens (numerical representation) that the model can process.</li>
  <li>GPT2LMHeadModel: A language model based on GPT-2. This is typically be used for text generation tasks.</li>
  <li>TrainingArguments: A class that holds the training configurations like batch size, learning rate, and number of epochs.</li>
   <li>Trainer: A high-level API that simplifies training, evaluation, and prediction of models in the Hugging Face library.</li>
    <li>AutoModelForSequenceClassification: A model class for sequence classification tasks (e.g., text classification).</li>
</ul>


<h3>Tokenizer and Model Setup</h3> 
<b>
</b>
<img src="https://imgur.com/7PmGOQG.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<ul>
  <li> tokenizer = GPT2Tokenizer.from_pretrained('gpt2'): This loads the pre-trained GPT-2 tokenizer from the Hugging Face model hub. The tokenizer will be used to convert raw text into tokenized input that the model can understand.</li>
  <li> model = AutoModelForSequenceClassification.from_pretrained('gpt2'): This loads a GPT-2 model, but specifically configured for sequence classification (as opposed to its default generative language modeling setup). It essentially takes the pre-trained GPT-2 model and adjusts it for classification tasks.</li>
  
</ul>


<h3>Tokenization and Model Inference</h3> 
<b>
</b>
<img src="https://imgur.com/SXcSHi4.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<ul>
  <li> text = "Replace me by any text you'd like.": A sample input text that will be used for tokenization and feeding to the model.</li>
  <li> encoded_input = tokenizer(text, return_tensors='pt'): This tokenizes the input text (text) and returns the result as a PyTorch tensor (the model requires tensors for input). The argument return_tensors='pt' ensures that the output is a PyTorch tensor, which is suitable for the model.</li>
  <li> output = model(**encoded_input): The tokenized input is fed to the model. **encoded_input unpacks the tokenized input as keyword arguments (e.g., input_ids, attention_mask), which are then passed into the model for inference. The output will contain the model’s predictions or logits.</li>
</ul>


<h3>Padding Token Adjustment</h3> 
<b>
</b>
<img src="https://imgur.com/zuWb6eh.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<ul>
  <li> This adjusts the tokenizer and model to use the end-of-sequence (EOS) token as the padding token. This is typically done when the model was not originally trained to handle padding tokens, so we make the EOS token serve both as the EOS marker and the padding token.
</li>
</ul>


<h3>Loading The Dataset</h3> 
<b>
</b>
<img src="https://imgur.com/CBEQjpW.png" height="60%" width="60%" alt="Disk Sanitization Steps"/>

<ul>
  <li> from datasets import load_dataset: This imports the load_dataset function from the datasets library, which is part of Hugging Face’s ecosystem.</li>
  <li> - ds = load_dataset("stanfordnlp/imdb"): This loads the IMDB dataset (a movie review dataset for sentiment classification) from the Hugging Face dataset hub. It provides a train and test split, along with the associated labels (positive or negative sentiment).</li>
  
</ul>


<h3>Tokenization Function For The Dataset</h3> 
<b>
</b>
<img src="https://imgur.com/GmyXjhk.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<ul>
  <li>This function takes an input of examples (which would be a batch of text data) and applies tokenization. It uses the tokenizer to tokenize the text, ensuring that all sequences are padded to a maximum length (padding="max_length") and truncated if they exceed that length (truncation=True).</li>
</ul>


<h3> Applying Tokenization to The Dataset</h3> 
<b>
</b>
<img src="https://imgur.com/c0T965I.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<ul>
  <li> dataset = load_dataset("imdb"): This loads the IMDB dataset again, but this time it is being stored in the dataset variable.</li>
   <li> tokenized_datasets = dataset.map(tokenize_function, batched=True): The map function applies the tokenize_function to the entire dataset. The batched=True argument ensures that the function processes the dataset in batches, which improves efficiency.</li>
</ul>


<h3> Setting Up Training Arguments</h3> 

<img src="https://imgur.com/il4iQU0.png" height="40%" width="40%" alt="Disk Sanitization Steps"/>
<ul>
<li>TrainingArguments: This is a configuration class for the training process.</li> 
<li> per_device_train_batch_size=8: The batch size for training on each device (e.g., GPU). It specifies how many samples will be processed per step.</li>
  <li> output_dir="./results": The directory where the training results (e.g., model checkpoints) will be saved.</li>
   <li> learning_rate=2e-5: The learning rate for the optimizer. It controls how much the model's weights will be updated during training.</li>
  <li> num_train_epochs=3: The number of times the model will go through the entire training dataset.</li>
</ul>


<h3> Setting Up The Trainer</h3> 

<img src="https://imgur.com/ge9JhOy.png" height="50%" width="50%" alt="Disk Sanitization Steps"/>
<b>Trainer: This is a high-level API that takes care of the training loop, including forward passes, backpropagation, and evaluation. The following arguments are passed to it:
</b>
<br />
<br />
<ul>
<li> model=model: The model to be trained.</li> 
<li> args=training_args: The training configuration, which includes parameters like batch size, learning rate, and number of epochs.</li>
<li> train_dataset=tokenized_datasets["train"]: The tokenized training dataset.
</li>
<li> eval_dataset=tokenized_datasets["test"]: The tokenized evaluation (test) dataset. The trainer will evaluate the model's performance on this dataset during training.</li>
</ul>


<h3>Training The Model</h3> 

<img src="https://imgur.com/OA4k78B.png" height="30%" width="30%" alt="Disk Sanitization Steps"/>
<ul>
<li> trainer.train(): This starts the training process using the configurations set earlier. The model will train on the training dataset for the specified number of epochs, evaluating itself on the test dataset after each epoch (depending on the Trainer settings).</li> 
</ul>


<h3>Evaluating The Model</h3> 
<b>The initial evaluation of the model returned the following results:
</b>
<br />
<br />
<img src="https://imgur.com/0unTIve.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>


<h3>Downloading Model and Tokenizer Files</h3> 

<img src="https://imgur.com/m3v9Uph.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<b>These lines show the model and tokenizer files being downloaded from the Hugging Face model hub. The downloads include the following files:
</b>
<br />
<br />
<ul>
<li> tokenizer_config.json: Contains configuration details for the tokenizer, such as special tokens and settings.</li> 
<li> vocab.json: The vocabulary file used by the tokenizer. It maps tokens (words or subwords) to unique integer IDs.</li>
<li> merges.txt: This file contains information about byte pair encoding (BPE) merges. BPE is a method used in tokenization that helps split words into subword units.</li>
<li>tokenizer.json: A more complete tokenizer configuration file that includes everything needed to recreate the tokenizer's behavior.</li>
<li> config.json: Configuration file for the model. This file contains model-specific parameters like the architecture (e.g., GPT-2, number of layers, hidden size, etc.)</li>
<li> model.safetensors: This is the model's weight file. It's where the pre-trained parameters of the model are stored.</li>
</ul>


<h3>Dataset Downloads</h3> 

<img src="https://imgur.com/tVmaOAi.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<b>These lines show that the program is downloading the IMDb dataset from the Hugging Face datasets library. The dataset files (including the training and test data) are being downloaded.
</b>
<br />
<br />
<ul>
<li>Each line represents a file being downloaded, including the dataset itself (the 21MB and 20.5MB files) and possibly metadata like the readme file.</li> 
</ul>


<h3>Data Generation (Splits)</h3> 

<img src="https://imgur.com/oD6wzrN.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<b>The program is now preparing the IMDB dataset splits (train, test, and unsupervised) for training. The dataset has been split into:
</b>
<br />
<br />
<ul>
<li> Train split: 25,000 examples for training the model.</li>
<li> Test split: 25,000 examples for evaluating the model.</li>
<li> Unsupervised split: 50,000 examples for unsupervised tasks (though this is not used for supervised training here).</li> 
</ul>


<h3>Data Mapping</h3> 

<img src="https://imgur.com/BmcWE4y.png" height="60%" width="60%" alt="Disk Sanitization Steps"/>

<ul>
<li>These lines show that the program is applying the tokenization function (defined earlier in the code) to the dataset. The map function is applying the tokenizer to each example in the training, test, and unsupervised splits.</li>
<li> - The 0%| indicates the current progress of processing the dataset. The actual tokenization process (splitting the text into tokens and ensuring proper padding/truncation) is happening here.</li>
</ul>

<h3>PHASE 2: PERFORMING PARAMETER-EFFICIENT FINE-TUNING</h3>
<b>The objective was to create a PEFT model from my already loaded model, run a training loop, and save the PEFT model weights.
</b>
<br />
<br />
<img src="https://imgur.com/FQBn3m4.png">


<h3>Importing LoraConfig</h3> 

<img src="https://imgur.com/qGQaGj6.png" height="30%" width="30%" />

<ul>
<li> from peft import LoraConfig: This imports the LoraConfig class from the peft library. The peft library is designed for parameter-efficient fine-tuning (PEFT) of models. LoraConfig specifically refers to the configuration needed for applying Low-Rank Adaptation (LoRA), a technique for efficiently fine-tuning large language models by adding low-rank adaptations to the pre-trained model.</li>
<li> config = LoraConfig(): This creates an instance of LoraConfig. The configuration object will hold settings that define how the LoRA adaptation is applied to the model, such as the rank of the matrices, learning rates, and other hyperparameters related to LoRA.</li> 
</ul>


<h3>Importing GPT-2 Model and Tokenizer</h3> 

<img src="https://imgur.com/SX5LQE2.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>

<ul>
<li> from transformers import ...: This imports various classes and functions from the Hugging Face transformers library.
</li> 
<li> GPT2Tokenizer: A tokenizer that is specifically designed for GPT-2. It converts raw text into tokens that GPT-2 can understand.</li>
<li> GPT2LMHeadModel: A class for GPT-2 used for language modeling tasks. This is not used in the current script directly, but it's imported here because it's part of the typical GPT-2 pipeline.</li>
<li> TrainingArguments and Trainer: These are classes for specifying the configuration for training (not used in the current code, but typically used for fine-tuning the model).</li>
<li> AutoModelForSequenceClassification: This class automatically loads the appropriate model architecture for sequence classification tasks (e.g., sentiment analysis). Here, it is used to load the GPT-2 model adapted for classification (instead of language modeling).</li>
<li> tokenizer = GPT2Tokenizer.from_pretrained('gpt2'): Loads the pre-trained GPT-2 tokenizer from the Hugging Face model hub.</li>
<li> model = AutoModelForSequenceClassification.from_pretrained('gpt2'): Loads the pre-trained GPT-2 model and adapts it for sequence classification. The model will have a classification head on top of the base GPT-2 transformer layers.</li>
</ul>


<h3>Tokenizing The Input Text</h3> 

<img src="https://imgur.com/SXcSHi4.png" height="80%" width="80%" />

<ul>
<li> text = "Replace me by any text you'd like.": This is a placeholder text that will be tokenized and passed to the model.</li>
<li> encoded_input = tokenizer(text, return_tensors='pt'): The text is tokenized using the GPT-2 tokenizer, and the result is returned as PyTorch tensors (return_tensors='pt'). This means that the tokenized input will be compatible with PyTorch models.</li>
<li> output = model(**encoded_input): The tokenized input (encoded_input) is passed to the model, and the model processes it to produce an output (e.g., logits for classification). The **encoded_input syntax unpacks the dictionary so that the model receives the tokenized values as keyword arguments (like input_ids, attention_mask).</li>

</ul>


<h3>Setting Padding Token</h3> 

<img src="https://imgur.com/cEZuKiY.png" height="80%" width="80%" />

<ul>
<li> tokenizer.pad_token = tokenizer.eos_token: Since GPT-2 does not have a dedicated padding token by default, this line sets the padding token to be the same as the end-of-sequence (EOS) token. This ensures that sequences are padded with the same token that marks the end of a sequence.</li>
<li> model.config.pad_token_id = model.config.eos_token_id: This sets the model’s configuration to use the EOS token's ID as the padding token's ID. This is necessary to align the model's configuration with the tokenizer.</li> 
</ul>


<h3>Loading The IMDB Database</h3> 

<img src="https://imgur.com/8EdWQbV.png" height="40%" width="40%" />

<ul>
<li>  from datasets import load_dataset: Imports the load_dataset function from the Hugging Face datasets library, which makes it easy to load common datasets (e.g., IMDb, SQuAD).</li>
<li> ds = load_dataset("stanfordnlp/imdb"): Loads the IMDb dataset from the Hugging Face datasets library. The IMDb dataset is commonly used for sentiment analysis and consists of movie reviews labeled as either positive or negative.</li> 
</ul>


<h3>Applying LORA to The Model</h3> 

<img src="https://imgur.com/QdmLYuB.png" height="40%" width="40%" />

<ul>
<li> from peft import get_peft_model: This imports the get_peft_model function from the peft library. This function is responsible for modifying a pre-trained model to apply parameter-efficient fine-tuning (PEFT) techniques like LoRA.</li>
<li> lora_model = get_peft_model(model, config): This applies LoRA to the pre-trained model (the GPT-2 model with the classification head). The config object contains the settings for the LoRA adaptation (like rank, learning rate, etc.). The resulting lora_model is now a modified version of the GPT-2 model that incorporates LoRA for parameter-efficient fine-tuning.</li> 
</ul>


<h3>Saving The LORA-Modified Model</h3> 

<img src="https://imgur.com/bBGu6kY.png" height="50%" width="50%" />

<ul>
<li> lora_model.save_pretrained("gpt-lora"): This saves the LoRA-adapted model (lora_model) to the directory gpt-lora. This allows you to easily reload the modified model later or share it with others. It saves both the model weights and the configuration.</li>

</ul>


<h3>Printing Trainable Parameters of LORA Model</h3> 

<img src="https://imgur.com/o16LXyn.png" height="50%" width="50%" />

<ul>
<li> lora_model.print_trainable_parameters(): This function prints the number of trainable parameters in the LoRA-adapted model. It’s useful for understanding how many parameters are being fine-tuned when applying PEFT techniques like LoRA. This will typically be much fewer than the total number of parameters in the original model, as LoRA only modifies a small number of parameters (those related to the low-rank adaptation).</li>

</ul>


<h3>Trainable Parameters and Total Parameters</h3> 

<b>These are the training results that were returned from the PEFT model.
</b>
<br />
<br />
<img src="https://imgur.com/vkG2PnS.png" height="80%" width="80%" alt="Disk Sanitization Steps"/>
<br />
<b>This line gives an overview of the model's trainable parameters compared to the total number of parameters.
</b>
<br />
<ul>
<li> Trainable params: 294,912 — These are the parameters that will be updated during training. These usually correspond to the parts of the model that are being fine-tuned (in this case, it includes the LoRA-adapted parameters and possibly the newly initialized classification head).</li>
<li> All params: 124,736,256 — This is the total number of parameters in the model, including both trainable and non-trainable parameters (such as the frozen pre-trained layers of the GPT-2 model).</li>
<li> Trainable %: 0.2364% — This percentage indicates that only about 0.24% of the total parameters in the model are being fine-tuned (trainable). This is because LoRA only adds a small number of additional parameters compared to the original model size. This is one of the advantages of LoRA — it allows you to fine-tune large models like GPT-2 with a very small number of additional parameters.</li> 
</ul>


<h3>PHASE 3: PERFORMING INFERENCE WITH A PEFT MODEL</h3>
<b>The objective for this stage of the process was to load the PEFT model weights and evaluate the performance of the trained PEFT model and to compare the results to the results prior to fine-tuning.
</b>
<br />
<br />

<img src="https://imgur.com/d01MfPY.png" height="80%" width="80%" />
<b>Much of the code shown above has already been defined in earlier parts of this project, but the factors below summarizes what was accomplished in this stage:
</b>
<br />
<ul>
<li>The code sets up a model (GPT-2) for fine-tuning on a sequence classification task, with LoRA applied for efficient parameter updating.</li>
<li>It tokenizes a sample text and prepares a dataset for training.</li>
<li>The model is then fine-tuned on the IMDb dataset with specified training parameters, using Hugging Face's Trainer class for managing the training loop.</li> 
</ul>
<b>This code effectively combines state-of-the-art techniques like LoRA and Hugging Face's transformers for efficient fine-tuning.
</b>
