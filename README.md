# LLM-catalog
Large Language Models (and not only) summarized in a table

|model|year|paper|model type / objective|short info|parameters|training corpora|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|-|2015|[Dai et al.](https://arxiv.org/abs/1511.01432) (Google)|RNN (LSTM)|idea of fine-tuning pre-trained domain-specific language models|-|IMDB, DBPedia, Newsgroups|
|**Transformer**|2017|[Vaswani et al.](https://arxiv.org/abs/1706.03762) (Google)|seq2seq for machine translation|original Transformer architecture|up to 213M|WMT (translation dataset)|
|ULMFiT|2018|[Howard & Ruder](https://arxiv.org/abs/1801.06146) (fast.ai)|RNN (AWD-LSTM)|idea of fine-tuning pre-trained general-domain language models|-|Wikitext-103|
|ELMo|2018|[Peters et al.](https://arxiv.org/abs/1802.05365) (Allen Institute for AI)|bidirectional RNN|embeddings from LM added as input to other task-specific models|94M|1B Word LM Benchmark|
|**GPT**|[2018](https://openai.com/blog/language-unsupervised)|[Radford et al.](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (OpenAI)|autoregressive|first LLM using Transformer model (decoder)|110M|BooksCorpus|
|**BERT**|[2018](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)|[Devlin et al.](https://arxiv.org/abs/1810.04805) (Google)|masked LM + next sentence prediction|idea of masked language modeling (bidirectional encoder)|110M/340M|BooksCorpus + Wikipedia|
|**Transformer-XL**|2019|[Dai et al.](https://arxiv.org/abs/1901.02860) (CMU + Google)|masked LM + next sentence prediction|beyond fixed-length context (processing segments)|?|Wikitext-103, 1B Word LM Benchmark|
|**XLM**|2019|[Lample & Conneau](https://arxiv.org/abs/1901.07291) (Facebook)|autoregressive or MLM|cross-lingual language models|570M|Wikipedia, MultiUN, OPUS|
|**GPT-2**|[2019](https://openai.com/blog/better-language-models)|[Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (OpenAI) |autoregressive|first model to surpass 1B parameters|1.5B|WebText (OpenAI internal, 40GB)|
|**ERNIE**|2019|[Zhang et al.](https://arxiv.org/abs/1905.07129) (Tsinghua University)|masked LM + denoising autoencoder|text encoder + knowledge graph|114M|Wikipedia + Wikidata|
|**XLNet**|2019|[Yang et al.](https://arxiv.org/abs/1906.08237) (CMU + Google)|permutation LM|idea of permutation language modeling|340M|BooksCorpus + Wikipedia + Giga5 + ClueWeb + CommonCrawl|
|**RoBERTa**|2019|[Liu et al.](https://arxiv.org/abs/1907.11692) (Facebook)|masked LM|modifications to BERT after ablation study|355M|BooksCorpus + Wikipedia + CC-News + OpenWebText + Stories, 160 GB|
|**Megatron-LM**|2019|[Shoeybi et al.](https://arxiv.org/abs/1909.08053) (NVIDIA)|autoregressive or MLM|even larger multi-billion parameter models based on GPT/BERT|8.3B|Wikipedia + CC-Stories + RealNews + OpenWebText|
|**ALBERT**|2019|[Lan et al.](https://arxiv.org/abs/1909.11942) (Google)|masked LM + sentence order prediction|reduced #params by embedding decomposition + cross-layer param sharing|up to 235M|same as BERT|
|**DistilBERT**|2019|[Sanh et al.](https://arxiv.org/abs/1910.01108) (Hugging Face)|masked LM + next sentence prediction|obtained from BERT via knowledge distillation (teacher-student)|66M|same as BERT|
|**T5**|2019|[Raffel et al.](https://arxiv.org/abs/1910.10683) (Google)|seq2seq|encoder-decoder pre-trained with unsupervised denoising objective, fine-tuned with multi-task objective (tasks formulated as text-to-text)|up to 11B|C4 (Colossal Clean Crawled Corpus), 750GB (stage 1); supervised datasets (stage 2)|
|**BART**|2019|[Lewis et al.](https://arxiv.org/abs/1910.13461) (Facebook)|seq2seq|pre-trained as a denoising autoencoder: to restore the corrupted input|BERT+10%|[same as BERT](https://github.com/facebookresearch/fairseq/issues/3550)|
|**XLM-RoBERTa**|2019|[Conneau et al.](https://arxiv.org/abs/1911.02116) (Facebook)|masked LM|multi-lingual model pre-trained on texts in 100 languages|550M|CommonCrawl in 100 languages|
|**Meena**|2020|[Adiwardana et al.](https://arxiv.org/abs/2001.09977) (Google)|seq2seq (for dialogue)|multi-turn chatbot trained to minimize perplexity of the next token|2.6B|public domain social media conversations|
|**Turing NLG**|[2020](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft)|- (only blogpost) (Microsoft)|autoregressive|a language model scaled up to 17B parameters|17B|"same type of data that Megatron-LM models were trained on"|
|**ELECTRA**|2020|[Clark et al.](https://arxiv.org/abs/2003.10555) (Stanford + Google)|replaced token detection|GAN-like pre-training; generator corrupts the input, discriminator detects corrupted tokens|same as BERT|same as BERT, for largest model: same as XLNet|
|**GPT-3**|2020|[Brown et al.](https://arxiv.org/abs/2005.14165) (OpenAI)|autoregressive|very similar to GPT-2, but larger (175B params; largest at that time)|175B|CommonCrawl + extended WebText + Books + Wikipedia|
|**DeBERTa**|2020|[He et al.](https://arxiv.org/abs/2006.03654) (Microsoft)|masked LM|BERT with disentangled attention (word content and position separated) + enhanced mask decoder|up to 1.5B|Wikipedia + BooksCorpus + OpenWebText + Stories|
|**mT5**|2020|[Xue et al.](https://arxiv.org/abs/2010.11934) (Google)|seq2seq|multilingual T5 for 101 languages|up to 11B|CommonCrawl in 101 languages (mC4)|
|Switch Transformer|2021|[Fedus et al.](https://arxiv.org/abs/2101.03961) (Google)|seq2seq (Mixture of Experts)|sparsely-activated model / MoE - parameters (part of the model to be used) depend on the input data|1.6T (MoE)|same as in T5 and mT5|
|**GLM**|2021|[Du et al.](https://arxiv.org/abs/2103.10360) (Tsinghua University)|TBD|TBD|130B|TBD|
|**GPT-Neo**|[2021](https://www.eleuther.ai/research/projects/gpt-neo)|- (EleutherAI)|autoregressive|replication of the GPT-3 architecture (with much less parameters)|2.7B|The Pile|
|**GPT-J**|2021|- (EleutherAI)|autoregressive|replication of the GPT-3 architecture (with much less parameters); seems very similar to GPT-Neo|6B|The Pile|
|**Jurassic-1**|2021|[Lieber et al.](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) (AI21 Labs)|autoregressive|GPT-3 like with "optimized" depth-to-width ratio (shallower but wider?) and larger vocabulary|178B|attempt to replicate GPT-3 data using publicly available data|
|**FLAN**|2021|[Wei et al.](https://arxiv.org/abs/2109.01652) (Google)|autoregressive|137B LaMDA-PT model fine-tuned on instructions|137B|a mixture of 62 NLU and NLG tasks (see paper for details)|
|**T0**|2021|[Sanh et al.](https://arxiv.org/abs/2110.08207) (Hugging Face)|seq2seq|T5 model fine-tuned on a large mixture of supervised tasks with a unified prompt format|11B|[P3](https://huggingface.co/datasets/bigscience/P3)|
