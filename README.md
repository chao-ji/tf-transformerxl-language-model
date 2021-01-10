# TransformerXL for Language Modeling 

TransformerXL is a recent neural network architecture (based on the original [Transformer model](https://arxiv.org/abs/1706.03762)) designed for language modeling -- the task to predict which word comes next given a sequence of past words. Unlike purely recurrent models (like LSTM) that squash the information of all past words into a single fixed-size embedding vector, TransformerXL bases its computation of the likelihood of all possible "next words" on a "memory block", a finite sequence of hidden states from previous words, each of which is in turn computed from their own memory blocks. Combined with the self-attention mechanism, it makes it possible to draw information from a much extended range of context to generate texts that are coherent and on-topic. 

This is a tensorflow implementation of TransformerXL that features

* Pipelines for training, evaluation, and inference (i.e. generating text based a piece of "prompt" text).
* Regular Embedding Layer (i.e. matrix dot product is used to convert token IDs to embedding vectors), as well as the "Adaptive Embedding Layer" (and also Adaptive Softmax Layer), which is an algorithmic technique to reducing the computational cost of neural language modeling with very large vocabulary.
* Multiple decoding method for text generation -- Top-K, nucleus sampling, as well as Beam search.
* Both subword and whole word tokenizer.


## Installation
You can clone this repository by running

```bash
git clone git@github.com:chao-ji/tf-transformerxl.git
```

The clone & update the submodule by running
```bash
cd tf-transformer
git submodule update --init --recursive
```

## Data Preparation

We first need to tokenize the raw text into word or subword tokens, from which a vocabulary is defined, then encode the tokenized sequence of words/subwords into IDs. The token ids are then split up into fixed-length segments and packed in batches, and serialized into `.tfrecord` files.

To convert training corpus `/path/to/train.txt` into `.tfrecord` file, run 

```bash
python create_tfrecord_language_model.py \
  --filenames=/path/to/train.txt \
  --subword=False \
  --batch_size=32 \
  --seq_len=224 \
  --vocab_name=vocab_file \
  --output_filename=word_train \
```
This generates `word_train.tfrecord`, as well as `word_train.json`, a configuration file storing meta-data. `vocab_file` is a text file storing the list of tokens from `/path/to/train.txt`.

Likewise, to convert validation corpus `/path/to/valid.txt` into `.tfrecord` file, run

```bash
python create_tfrecord_language_model.py \
  --filenames=/path/to/valid.txt \
  --subword=False \
  --batch_size=32 \
  --seq_len=224 \
  --vocab_name=vocab_file \
  --output_filename=word_valid \
  --use_exist_vocab
```

The flag `--use_exist_vocab` says that we'd like to reuse the vocab file `vocab_file` to restore the vocabulary generated for the training corpus to tokenize the validation corpus.

Note that `--subword` can be turned on or off for subword or whole word tokenization.

## Training
To train a model, run

```bash
python run_trainer.py \
  --filename=/path/to/word_train \
  --vocab_path=/path/to/vocab_file \
  --model_dir=. \
  --adaptive_embedding=True
```
where `/path/to/word_train` and `/path/to/vocab_file` are prefixes to the `.tfrecord` file and vocabulary file in the data preparation part, and the checkpoint files storing trained model weights will be saved under the current directory (i.e. `--model_dir=.`)

Note when `--adaptive_embedding` is turned on, we use **adaptive input/softmax** which are algorithmic techniques for optimizing the ID-to-embedding and embedding-to-logits computation for very-large vocabulary. 

For detailed usage, run 
```bash
python run_trainer.py --help
```

## Evaluation
Language models are evaluated in terms of Perplexity that quantifies how "confused" the model is when asked to predict which word comes next (so smaller numbers are better). 

To evaluate a pretrained model, run

```bash
python run_evaluator.py \   
  --filename=/path/to/word_valid \
  --vocab_path=/path/to/vocab_file \
  --model_dir=. \
  --adaptive_embedding=True
```

`/path/to/word_valid` is the prefix to the `.tfrecord` file, and `/path/to/vocab_file` is the same vocabulary file used for training.

## Prompted Text Generation

To generate a piece of text using a trained TransformerXL model, run
```bash
python run_inferencer.py \
  --prompt_filename=/path/to/prompt.txt \
  --filename=/path/to/word_train \
  --vocab_path=/path/to/vocab_file \
  --model_dir=. \
  --adaptive_embedding=True \
  --decoding_method=nucleus 
```

`/path/to/prompt.txt` is the path to the text file storing a piece of text as the "prompt", from which a sequence of words that follow is generated autoregressively . You can choose to use `nucleus` (Nucleus Sampling), `topk` (Top-k Sampling), or `beam_search` as the decoding method.


## Examples

### WikiText103 dataset

We train a TransformerXL model on WikiText103 dataset using default hyperparameters (sequnece length = memory length = 224, num layers = 9, hidden size = 512, batch size = 32, w/ whole-word tokenization and vocabulary size around 270k, which is the largest model that can be trained on 11 GB of GPU memory, with cosine decay for learning rate, and trained for 400k iterations with initial learning rate 2.5e-4), giving rise to perplexity 20.43 and 21.61 on the validation and test set, respectively. 

Given the prompt sequence as follows:

`= Robert Boulter = Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris <unk> , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . Boulter starred in the 2011 film Mercenaries directed by Paris <unk> . = = Career = = = = = 2000 – 2005 = = = In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed " Scott Parry " in the episode , " In Safe Hands " . Boulter starred as " Scott " in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . A review of Boulter 's performance in The Independent on Sunday described him as " horribly menacing " in the role , and he received critical reviews in The Herald , and Evening Standard . He appeared in the television series Judge John Deed in 2002 as " <unk> Armitage " in the episode " Political <unk> " , and had a role as a different character " Toby Steele " on The Bill . He had a recurring role in 2003 on two episodes of The Bill , as character " Connor Price " . In 2004 Boulter landed a role as " Craig " in the episode " Teddy 's Story " of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as " Darren " , in the 2005 theatre productions of the Philip Ridley play Mercury Fur . It was performed at the Drum Theatre in Plymouth , and the <unk> Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : " The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . " The Guardian noted , " Ben Whishaw and Robert Boulter offer tenderness amid the savagery . " = = = 2006 – present = = = In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / <unk> / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : " I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , <unk> and Citizenship at the National . He played my brother in Mercury Fur . " He portrayed " Jason Tyler " on the 2006 episode of the television series , Doctors , titled " Something I Ate " . Boulter starred as " William " in the 2007 production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , " Robert Boulter brings a touching vulnerability to the stage as William . " Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris <unk> , and Donkey Punch directed by Olly Blackburn . Boulter portrayed a character named " Sean " in Donkey Punch , who tags along with character " Josh " as the " quiet brother ...`

using `nucleus` as the decoding method, the generated text are:

`who has already killed Luke " in a variety of films , in Soldier Soldier , and was posthumously nominated for an Emmy Award . <eos> <eos> = = = Listener = = = <eos> <eos> In 2006 , Udall signed a contract with Magnolia Publishing to publish a children 's book called The Last Flute , now jointly written by James and John infallibly . The book was published in 2007 and has previously been published by <unk> . In January 2008 , Magnolia released a companion book , I 'll Make Your Own Angel , which has been expanded with a book by Duncan Smith , the playwright and poet whose real life brother would later appear on the serial . The book was commercially successful and attracted media attention from many concerned educators and novels because it would be able to create and generate a " significant imagination " for the show 's audience . Magnolia has also also produced The Fragmentary Allergic , for which bacteriostatic fosters the production of mosquito proteins from infected infected pets . <eos> The book has also been published by Nursery Works in 2011 . Nursery Works has also published a manga series entitled The Fragmentary Allergic , written by Louisa Trotter and illustrated by Judy Hucknall . <eos> <eos> = = Reception = = <eos> <eos> In December 2006 , a reporter for the Los Angeles Times reported that Magnolia had received the most reviews , with an average of 18 @.@ 8 out of 20 . This was based on the publication of the story and a three @-@ page illustration by John hemi of the Dundee Advertiser newspaper ; the three chapters were criticized by readers for being similar to the first chapter , but quite nicely shown in a higher light . It was placed eighth on the 2007 list of best six novels written by Priscilla Hiscock , based on the 2010 issue of The Guardian . Because it was the same magazine 's seventh book that was published by Nursery Works , Magnolia was placed seventh on the 2007 list of best selling books by the publishers , at number five and number four . Magnolia was also ranked second on Saturday Day 's Top 5 Most Influential People and five of California 's 19 most influential Christian writers of the 20th century . <eos> The hardcover printing by HarperCollins , published in October 2007 , became one of the most popular paperback books of the year . Macintosh Macintosh , Microsoft Windows , and PC had entered the top 50 best @-@ selling books of the year . <eos> <eos> <eos> = Undertaker = <eos> <eos> The Undertaker ( / <unk> / ; short for " Pure Fierce @-@ faced " ) was a powerful anti @-@ Shinto deity that made extensive use of divine power and the ability to work in the afterlife and Vishnu 's always @-@ sufficient level of power for such purposes . It is in direct contrast to the . <eos>`

and using `beam_search`, the generated text are:

`... " . The film was nominated for a Golden Globe Award for Best Actor – Motion Picture Musical or Comedy , but failed to win any of the awards . In 2009 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . The film was nominated for a Golden Globe Award for Best Actor – Motion Picture Musical or Comedy , but failed to win any of the awards . In 2010 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2011 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2012 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2013 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2014 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2014 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2015 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2015 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which was directed by David <unk> . In 2016 , he starred in the comedy @-@ drama The <unk> , which` 

It's clear that beam search gives rise to very repetitive and degenerate text, which is already a known problem related to the likelihood-maximizing nature of beam search, and is currently an active area of research.



## Reference
* [Official implementation from the authors](https://github.com/kimiyoung/transformer-xl)
* [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860), Dai *et al.* 2019
* [Attention is all you need](https://arxiv.org/abs/1706.03762), Vaswani *et al.* 2017.
* [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309), Grave *et al.* 2017
* [Adaptive Input Representations for Neural Language Modeling](https://arxiv.org/abs/1809.10853), Baevski *et al.* 2018.
* [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman *et al.* 2019
