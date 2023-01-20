# LLM-catalog
Majority of the Large Language Models (and not only) summarized in a table. From the original Transformer to ChatGPT and beyond.

|model|year|paper|model type / objective|short info|parameters|training corpora|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|-|2015|[Dai & Le](https://arxiv.org/abs/1511.01432) (Google)|autoregressive or autoencoder using RNN (LSTM)|idea of pre-training domain-specific language models to be later fine-tuned|?|IMDB, DBPedia, 20 Newsgroups|
|**Transformer**|2017|[Vaswani et al.](https://arxiv.org/abs/1706.03762) (Google)|seq2seq for machine translation|original Transformer architecture|up to 213M|WMT 2014 (translation dataset)|
|ULMFiT|2018|[Howard & Ruder](https://arxiv.org/abs/1801.06146) (fast.ai)|autoregressive using RNN (AWD-LSTM)|idea of pre-training general-domain language models to be later fine-tuned|?|Wikitext-103|
|ELMo|2018|[Peters et al.](https://arxiv.org/abs/1802.05365) (Allen Institute for AI)|bidirectional LM using RNN (LSTM)|embeddings from LM added as input to other task-specific models|94M|1B Word LM Benchmark|
|**GPT**|[2018](https://openai.com/blog/language-unsupervised)|[Radford et al.](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (OpenAI)|autoregressive|first LLM using the Transformer model (decoder-only)|117M|BooksCorpus|
|**BERT** ([weights](https://huggingface.co/bert-base-uncased))|[2018](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)|[Devlin et al.](https://arxiv.org/abs/1810.04805) (Google)|masked LM + next sentence prediction|idea of masked language modeling (bidirectional encoder)|110M/340M|BooksCorpus + Wikipedia|
|**Transformer-XL**|2019|[Dai et al.](https://arxiv.org/abs/1901.02860) (CMU + Google)|masked LM + next sentence prediction|learning dependency beyond fixed-length context (processing segments)|up to ~0.8B|Wikitext-103, 1B Word LM Benchmark|
|**XLM**|2019|[Lample & Conneau](https://arxiv.org/abs/1901.07291) (Facebook)|autoregressive or masked LM|cross-lingual language models|570M|Wikipedia, MultiUN, OPUS|
|**GPT-2** ([weights](https://huggingface.co/gpt2))|[2019](https://openai.com/blog/better-language-models)|[Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (OpenAI) |autoregressive|first model to surpass 1B parameters|1.5B|WebText (OpenAI internal, 40GB)|
|**ERNIE**|2019|[Zhang et al.](https://arxiv.org/abs/1905.07129) (Tsinghua University)|masked LM + denoising autoencoder|text encoder + knowledge graph|114M|Wikipedia + Wikidata|
|**XLNet** ([weights](https://huggingface.co/xlnet-base-cased))|2019|[Yang et al.](https://arxiv.org/abs/1906.08237) (CMU + Google)|permutation LM|idea of permutation language modeling|340M|BooksCorpus + Wikipedia + Giga5 + ClueWeb + CommonCrawl|
|**RoBERTa** ([weights](https://huggingface.co/roberta-base))|2019|[Liu et al.](https://arxiv.org/abs/1907.11692) (Facebook)|masked LM|modifications to BERT after ablation study|355M|BooksCorpus + Wikipedia + CC-News + OpenWebText + Stories, 160 GB|
|**Megatron-LM**|2019|[Shoeybi et al.](https://arxiv.org/abs/1909.08053) (NVIDIA)|autoregressive or MLM|even larger multi-billion parameter models based on GPT/BERT|8.3B|Wikipedia + CC-Stories + RealNews + OpenWebText|
|**ALBERT** ([weights](https://huggingface.co/albert-base-v2))|2019|[Lan et al.](https://arxiv.org/abs/1909.11942) (Google)|masked LM + sentence order prediction|reduced #params by embedding decomposition + cross-layer param sharing|up to 235M|same as BERT|
|**DistilBERT** ([weights](https://huggingface.co/distilbert-base-uncased))|2019|[Sanh et al.](https://arxiv.org/abs/1910.01108) (Hugging Face)|masked LM + next sentence prediction|obtained from BERT via knowledge distillation (teacher-student)|66M|same as BERT|
|**T5**<br />([weights](https://huggingface.co/t5-base))|2019|[Raffel et al.](https://arxiv.org/abs/1910.10683) (Google)|seq2seq|encoder-decoder pre-trained with unsupervised denoising objective, fine-tuned with multi-task objective (tasks formulated as text-to-text)|up to 11B|C4 (Colossal Clean Crawled Corpus), 750GB (stage 1); supervised datasets (stage 2)|
|**BART** ([weights](https://huggingface.co/facebook/bart-base))|2019|[Lewis et al.](https://arxiv.org/abs/1910.13461) (Facebook)|seq2seq|pre-trained as a denoising autoencoder: to restore the corrupted input|BERT+10%|[same as BERT](https://github.com/facebookresearch/fairseq/issues/3550)|
|**XLM-RoBERTa** ([weights](https://huggingface.co/xlm-roberta-base))|2019|[Conneau et al.](https://arxiv.org/abs/1911.02116) (Facebook)|masked LM|multi-lingual model pre-trained on texts in 100 languages|550M|CommonCrawl in 100 languages|
|**Meena**|2020|[Adiwardana et al.](https://arxiv.org/abs/2001.09977) (Google)|seq2seq (for dialogue)|multi-turn chatbot trained to minimize perplexity of the next token|2.6B|public domain social media conversations|
|**Turing NLG**|[2020](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft)|- (only blogpost) (Microsoft)|autoregressive|a language model scaled up to 17B parameters|17B|"same type of data that Megatron-LM models were trained on"|
|**ELECTRA** ([weights](https://huggingface.co/google/electra-base-discriminator))|2020|[Clark et al.](https://arxiv.org/abs/2003.10555) (Stanford + Google)|replaced token detection|GAN-like pre-training; generator corrupts the input, discriminator detects corrupted tokens|same as BERT|same as BERT, for largest model: same as XLNet|
|**GPT-3** ([API](https://beta.openai.com/docs/model-index-for-researchers))|2020|[Brown et al.](https://arxiv.org/abs/2005.14165) (OpenAI)|autoregressive|very similar to GPT-2, but larger (175B params; largest at that time)|175B|CommonCrawl + extended WebText + Books + Wikipedia|
|**DeBERTa** ([weights](https://huggingface.co/microsoft/deberta-base))|2020|[He et al.](https://arxiv.org/abs/2006.03654) (Microsoft)|masked LM|BERT with disentangled attention (word content and position separated) + enhanced mask decoder|up to 1.5B|Wikipedia + BooksCorpus + OpenWebText + Stories|
|**mT5** ([weights](https://huggingface.co/google/mt5-base))|2020|[Xue et al.](https://arxiv.org/abs/2010.11934) (Google)|seq2seq|multilingual T5 for 101 languages|up to 11B|CommonCrawl in 101 languages (mC4)|
|*Switch Transformer*|2021|[Fedus et al.](https://arxiv.org/abs/2101.03961) (Google)|seq2seq (Mixture of Experts)|sparsely-activated model / MoE - parameters (part of the model to be used) depend on the input data|1.6T (MoE)|same as in T5 and mT5|
|**GLM** ([weights](https://huggingface.co/BAAI/glm-10b))|2021|[Du et al.](https://arxiv.org/abs/2103.10360) (Tsinghua University)|autoregressive blank infilling|TBD|up to 10B|TBD|
|**GPT-Neo** ([weights](https://huggingface.co/EleutherAI/gpt-neo-2.7B))|[2021](https://www.eleuther.ai/research/projects/gpt-neo)|- (EleutherAI)|autoregressive|replication of the GPT-3 architecture (with much less parameters)|2.7B|The Pile|
|**GPT-J** ([weights](https://huggingface.co/EleutherAI/gpt-j-6B))|2021|- (EleutherAI)|autoregressive|replication of the GPT-3 architecture (with much less parameters); seems very similar to GPT-Neo|6B|The Pile|
|**Jurassic-1** ([API](https://www.ai21.com/studio))|2021|[Lieber et al.](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) (AI21 Labs)|autoregressive|GPT-3 like with "optimized" depth-to-width ratio (shallower but wider?) and larger vocabulary|178B|attempt to replicate GPT-3 data using publicly available data|
|**FLAN**|2021|[Wei et al.](https://arxiv.org/abs/2109.01652) (Google)|autoregressive|137B LaMDA-PT model fine-tuned on instructions|137B|a mixture of 62 NLU and NLG tasks (see paper for details)|
|**T0** ([weights](https://huggingface.co/bigscience/T0))|2021|[Sanh et al.](https://arxiv.org/abs/2110.08207) (Hugging Face)|seq2seq|T5 model fine-tuned on a large mixture of supervised tasks with a unified prompt format|11B|[P3](https://huggingface.co/datasets/bigscience/P3) (Public Pool of Prompts|
|**Megatron-Turing NLG**|[2021](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model)|[Smith et al.](https://arxiv.org/abs/2201.11990) (Microsoft + NVIDIA)|autoregressive|largest model at that time, 3x larger than GPT-3|530B|a subset of The Pile + CommonCrawl + RealNews + CC-Stories|
|**RETRO**|2022|[Borgeaud et al.](https://arxiv.org/abs/2112.04426) (DeepMind)|seq2seq (+ retrieval)|input is split into chunks; for each chunk, nearest neighbor entries are retrieved from DB to improve modeling|up to 7B|multilingual MassiveText (see Gopher paper)|
|*GLaM*|2022|[Du et al.](https://arxiv.org/abs/2112.06905) (Google)|autoregressive (Mixture of Experts)|another MoE model, this time autoregressive, with over a trillion parameters|1.3T (MoE)|a mixture of webpages, conversations, forums, books, news|
|**Gopher**|2022|[Rae et al.](https://arxiv.org/abs/2112.11446) (DeepMind)|autoregressive|a family of language models (up to 280B) plus analysis of effect of model scaling|up to 280B|MassiveText (MassiveWeb + C4 + Books + News + Wiki + GitHub)|
|**LaMDA**|2022|[Thoppilan et al.](https://arxiv.org/abs/2201.08239) (Google)|autoregressive (for dialogue)|pre-trained on public dialogues and web documents, fine-tuned for safety and factual correctness (knowledge retrieval from external tools)|137B|publicly available dialogues and web documents (details in paper)|
|*ST-MoE*|2022|[Zoph et al.](https://arxiv.org/abs/2202.08906) (Google)|seq2seq (Mixture of Experts)|stable training of a large-scale sparse (Mixture of Experts) language model|269B (MoE)|mix of C4 corpus and dataset used for GLaM|
|**InstructGPT** ([API](https://beta.openai.com/docs/model-index-for-researchers))|2022|[Ouyang et al.](https://arxiv.org/abs/2203.02155) (OpenAI)|autoregressive|GPT-3 model trained to follow instructions using Reinforcement Learning with Human Feedback (RLHF)|175B|human demonstrations of desired model behavior for prompts (manually written + collected via OpenAI API)|
|**Chinchilla**|2022|[Hoffmann et al.](https://arxiv.org/abs/2203.15556) (DeepMind)|autoregressive|compute-optimal training; 4x smaller than Gopher but trained on 4x more data, beats larger models on many downstream tasks|70B|MassiveText (a different subset distribution than in Gopher)|
|**PaLM**|2022|[Chowdhery et al.](https://arxiv.org/abs/2204.02311) (Google)|largest model to date, efficiently trained using Google Pathways system|540B|based on datasets used in GLaM and LaMDA|
|**Anthropic assistant**|2022|[Bai et al.](https://arxiv.org/abs/2204.05862) (Anthropic)|autoregressive (for dialogue)|dialogue agent based on a language model trained with RLHF to be helpful and harmless|up to 52B|The Pile|
|**GPT-NeoX** ([weights](https://huggingface.co/EleutherAI/gpt-neox-20b))|[2022](https://www.eleuther.ai/research/projects/gpt-neox)|[Black et al.](https://arxiv.org/abs/2204.06745) (EleutherAI)|autoregressive|largest publicly available dense autoregressive model at that time|20B|The Pile|
|**OPT** ([weights](https://huggingface.co/facebook/opt-66b))|2022|[Zhang et al.](https://arxiv.org/abs/2205.01068) (Meta)|autoregressive|a family of language models (up to 175B) that (apart from the largest one) have publicly available weights|up to 175B|dataset from RoBERTa + The Pile + Reddit|
|**YaLM** ([weights](https://huggingface.co/yandex/yalm-100b))|2022|- (Yandex)|autoregressive|?|100B|?|
|**Atlas**|2022|[Izacard et al.](https://arxiv.org/abs/2208.03299) (Meta)|seq2seq (+ retrieval)|T5 language model + retrieval from a corpus of documents (joint pretraining)|up to 11B|Wikipedia, CommonCrawl|
|**Sparrow**|2022|[Glaese et al.](https://arxiv.org/abs/2209.14375) (DeepMind)|autoregressive (for dialogue)|dialogue agent based on Chinchilla LM trained with RLHF to be helpful and harmless, able to retrieve information from external source|70B|dialogue data collected by interaction with human annotators|
|**GLM-130B** ([weights](https://github.com/THUDM/GLM-130B))|2022|[Zeng et al.](https://arxiv.org/abs/2210.02414) (Tsinghua University)|autoregressive blank infilling|open bilingual 130B model for English and Chinese|130B|The Pile, LAMBADA|
|**Flan-T5** ([weights](https://huggingface.co/google/flan-t5-base)) & **Flan-PaLM**|2022|[Chung et al.](https://arxiv.org/abs/2210.11416) (Google)|seq2seq / autoregressive|T5 and PaLM models fine-tuned with instructions (FLAN-T5 weights released in several sizes)|up to 540B|a mixture of 1836 finetuning tasks from 4 sources (details in paper)
|**BLOOM** ([weights](https://huggingface.co/bigscience/bloom))|2022|[Le Scao et al.](https://arxiv.org/abs/2211.05100) (BigScience)|autoregressive|a 176B parameter model resulting from the BigScience collaboration (trained for 3.5 months in the first half of the year)|176B|ROOTS dataset (mix of natural and programming languages)|
|**BLOOMZ** ([weights](https://huggingface.co/bigscience/bloomz))|2022|[Muennighof et al.](https://arxiv.org/abs/2211.01786) (BigScience)|autoregressive|BLOOM finetuned on instructions|176B|TBD|
|**Galactica** ([weights](https://huggingface.co/facebook/galactica-6.7b))|2022|[Taylor et al.](https://arxiv.org/abs/2211.09085) (Meta)|autoregressive|a model trained on a corpus of scientific knowledge, performing strongly in knowledge-intensive scientific tasks|up to 120B|papers, textbooks, encyclopedias, code, knowledge bases etc.|
|**ChatGPT** ([API](https://chat.openai.com)|[2022](https://openai.com/blog/chatgpt)|- (only blogpost for now) (OpenAI)|autoregressive (for dialogue)|a model trained in a similar way as InstructGPT, using RLHF, in a dialogue/chat framework|?|human demonstrations of desired model behavior for prompts (see InstructGPT)|
