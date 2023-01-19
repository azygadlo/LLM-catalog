# LLM-catalog
Large Language Models (and not only) summarized in a table

|model|year|paper|model type / objective|short info|parameters|training corpora|company / institution|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|-|2015|[Dai et al.](https://arxiv.org/abs/1511.01432)|RNN (LSTM)|idea of fine-tuning pre-trained domain-specific language models|-|IMDB, DBPedia, Newsgroups|Google|
|**Transformer**|2017|[Vaswani et al.](https://arxiv.org/abs/1706.03762)|seq2seq for machine translation|original Transformer architecture|up to 213M|WMT (translation dataset)|Google|
|ULMFiT|2018|[Howard & Ruder](https://arxiv.org/abs/1801.06146)|RNN (AWD-LSTM)|idea of fine-tuning pre-trained general-domain language models|-|Wikitext-103|fast.ai|
|ELMo|2018|[Peters et al.](https://arxiv.org/abs/1802.05365)|bidirectional RNN|embeddings from LM added as input to other task-specific models|94M|1B Word LM Benchmark|Allen Institute for AI|
|**GPT**|[2018](https://openai.com/blog/language-unsupervised)|[Radford et al.](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)|autoregressive|first LLM using Transformer model (decoder)|110M|BooksCorpus|OpenAI|
|**BERT**|[2018](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)|[Devlin et al.](https://arxiv.org/abs/1810.04805)|masked LM + next sentence prediction|idea of masked language modeling (bidirectional encoder)|110M/340M|BooksCorpus + Wikipedia|Google|
|**Transformer-XL**|2019|[Dai et al.](https://arxiv.org/abs/1901.02860)|masked LM + next sentence prediction|beyond fixed-length context (processing segments)|?|Wikitext-103, 1B Word LM Benchmark|CMU + Google|
|**XLM**|2019|[Lample & Conneau](https://arxiv.org/abs/1901.07291)|autoregressive or MLM|cross-lingual language models|570M|Wikipedia, MultiUN, OPUS|Facebook|
|**GPT-2**|[2019](https://openai.com/blog/better-language-models)|[Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|autoregressive|first model to surpass 1B parameters|1.5B|WebText (OpenAI internal, 40GB)|OpenAI|
|**ERNIE**|2019|[Zhang et al.](https://arxiv.org/abs/1905.07129)|masked LM + denoising autoencoder|text encoder + knowledge graph|114M|Wikipedia + Wikidata|Tsinghua University|
|**XLNet**|[2019|[Yang et al.](https://arxiv.org/abs/1906.08237)|permutation LM|idea of permutation language modeling|340M|BooksCorpus + Wikipedia + Giga5 + ClueWeb + CommonCrawl|CMU + Google|
|**RoBERTa**|[2019|[Liu et al.](https://arxiv.org/abs/1907.11692)|masked LM|modifications to BERT after ablation study|355M|BooksCorpus + Wikipedia + CC-News + OpenWebText + Stories, 160 GB|Facebook|
