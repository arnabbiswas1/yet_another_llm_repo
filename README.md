# Yet Another LLM Repository

I am a Machine Learning Engineer/Software Developer exploring LLM for last few months. Here are the different resources on Large Language Model (LLM) which I am using or planning to refer to in future. This list doesn't include academic papers or theoretical courses. It mostly focuses on resources (code, tools, blogs, short videos) which will help an engineer to build applications around LLM. Please note, I am NOT doing AI; I am just building a layer around AI.

Not sure if you should explore LLM? Start with this highly motivational post :[The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer)

## What is LLM/GPT?
  - [Catching up on the weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms) ⭐⭐: First thing first. Lot of things have already happened around LLM. This post (or youtube video, if you prefer) by Simon Willson will help you in catching up from the end user perspective. 
  - [State of GPT by Andrej Karpathy - Microsoft Build ' 2023 Talk](https://www.youtube.com/watch?v=bZQun8Y4L2A) ⭐⭐: This talk captures most of the progress made from the technical perspective.
  - [Large language models, explained with a minimum of math and jargon](https://www.understandingai.org/p/large-language-models-explained-with) ⭐⭐: If you want understand how language model works (intutively), check this out. No harm, if you skip it.

## Getting Started
If you are already familiar with [ChatGPT](https://chat.openai.com/) and want to get started with LLM (to build an application around it or to understand its capabilities), you might want to try OpenAI Playground and APIs. On [signing up at OpenAI](https://platform.openai.com/signup), you will be allocated with certain amount of free quota (to use the service) which is more than sufficient to build your first prototype. Save your API Key.

- [Getting Started using OpenAI API](https://platform.openai.com/docs/quickstart) ⭐: This will in turn take you to [playground](https://platform.openai.com/playground). You don't need to write any code. Start here!
- [openai-cookbook](https://cookbook.openai.com/)⭐: Example code for accomplishing common tasks with OpenAI. If you are a developer, this is going to be your go-to place.

If you are located in US/UK, [Claude-2](https://claude.ai/login) by Anthropic is another option. As of Aug'2023, I have not used Claude, since it's not available in India.

Third option is to use Azure OpenAI. This is only available for Azure Enterprise Customers and Most Valuable Bloggers at this moment. You can request for it [here](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/limited-access).

## Online Courses
- [Short Courses by deeplearning.ai](https://www.deeplearning.ai/short-courses/) ⭐: If you are a developer, these super short courses, will help you get started immediately. Start with the [PromptEngineering course](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/), then based on your area of interest, go through other courses. You will need the OpenAI API Keys to try out the tutorials.
- [A Hackers' Guide to Language Models] by Jeremy Howard(https://www.youtube.com/watch?v=jkrNMKz9pWU) - 90 minutes introduction to LLM ⭐
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) ⭐: A bunch of session recorded in April 2023. Highly recommended.

## Prompt Engineering Resources
- [GPT best practices by OpenAI](https://platform.openai.com/docs/guides/gpt-best-practices) ⭐
- [Prompt engineering techniques by Azure](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions) ⭐
- [Prompting guides suggested by openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/related_resources.md#prompting-guides)

My suggestion would be not to spend lot of time going through different prompt engineering courses and guides. "GPT best practices by OpenAI" is more than enough. Browse through other resources on the need basis.

## Tools
There are a bunch of Open Source libraries, frameworks that have been developed around LLM. Here are few which I have used:

  - [Langchain](https://langchain.readthedocs.io/) ⭐
  - [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/)
  - [gradio](https://gradio.app/creating-a-chatbot/)
  - [Suggestions by openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/related_resources.md#prompting-libraries--tools)

## What are different LLMs available and how ?
GPT's by OpenAI are not the only LLMs available. There are other commercial and Open Source models which you might be interested in. Following are the leaderboards to compare the performance of various LLMs:

  - [Leaderboard by lmsys](https://chat.lmsys.org/?leaderboard) ⭐
  - [FastEval](https://fasteval.github.io/FastEval/) ⭐
  - [Open Source LLM Leaderboard by huggingface](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) ⭐
  - [MMLU Leaderboard](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)
  - [AlpacaEval: An Automatic Evaluator for Instruction-following Language Models](https://tatsu-lab.github.io/alpaca_eval/)

## Use Cases for LLM
langchain documentation captures most of the popular use cases: [link](https://docs.langchain.com/docs/category/use-cases)

## Use Case : Retrieval-Augmented Generation / Document QA
Probably the most popular application of LLM: Build a ChatBot on your private data. But remember, semantic based search doesn't always work  

#### Blogs:
  - [Grounding LLMs](https://techcommunity.microsoft.com/t5/fasttrack-for-azure/grounding-llms/ba-p/3843857) ⭐
  - [Embedding-based retrieval alone might be insufficient](https://eugeneyan.com/writing/llm-experiments/#embedding-based-retrieval-alone-might-be-insufficient) ⭐

#### Courses
  - [LLM Bootcamp: Augmented Language Models](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/augmented-language-models/) ⭐
  - [deeplearning.ai: LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
  - [deeplearning.ai: Large Language Models with Semantic Search](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)

#### Code Examples
  - [openai-cookbook: Question answering over your data using embedding](https://cookbook.openai.com/examples/question_answering_using_embeddings) ⭐
  - [Question answering using a search API and re-ranking](https://cookbook.openai.com/examples/question_answering_using_a_search_api)

#### Tools/Frameworks
  - [Langchain](https://langchain.readthedocs.io/) ⭐
  - [LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) ⭐

#### Case Study
  - [Ask like a human: Implementing semantic search on Stack Overflow](https://stackoverflow.blog/2023/07/31/ask-like-a-human-implementing-semantic-search-on-stack-overflow/)

## Use Case : Agents
  - [Andrej Karpathy on Why you should work on AI AGENTS!](https://www.youtube.com/watch?v=fqVLjtvWgq8) ⭐⭐

#### Blog

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) ⭐

#### Courses

  - [LLM Bootcamp: Harrison Chase's talk on Agents](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/chase-agents/)

#### Code Examples

  - [openai-cookbook: How to call functions with chat models](https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models)
  - [openai-cookbook: How to build a tool-using agent with LangChain](https://cookbook.openai.com/examples/how_to_build_a_tool-using_agent_with_langchain)
  - [Agents Documentation by langchain](https://docs.langchain.com/docs/components/agents/)

## Automatic Speech Recognition
  - whisper by openai ([openai doc](https://platform.openai.com/docs/guides/speech-to-text), [github](https://github.com/openai/whisper), [model-card](https://github.com/openai/whisper/blob/main/model-card.md))
  - [Massively Multilingual Speech by facebook research](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) (CC-BY-NC 4.0)
  - [SeamlessM4T](https://github.com/facebookresearch/seamless_communication)

## Evaluation of LLMs/LLM based systems
The way we evaluate a LLM and a system built using LLM are different. Here I am going to focus mostly on evaluation methodologies used for LLM based systems

#### Blogs/Courses

- [Evaluating LLM-based Applications: Workshop by Josh Tobin](https://www.youtube.com/watch?v=r-HUnht-Gns) ⭐
- [How to Evaluate, Compare, and Optimize LLM Systems - Weights & Biases](https://wandb.ai/ayush-thakur/llm-eval-sweep/reports/How-to-Evaluate-Compare-and-Optimize-LLM-Systems--Vmlldzo0NzgyMTQz) ⭐
- [Using LLMs To Evaluate LLMs](https://medium.com/arize-ai/using-llms-to-evaluate-llms-c69da454048c)

#### Tools
  - [Azure Promptflow](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow?view=azureml-api-2): Service to evaluate and track LLM Experiments
  - [openai-evals](https://github.com/openai/evals): A framework for evaluating LLMs (large language models) or systems built using LLMs as components
  - [langchain-evals](https://python.langchain.com/docs/guides/evaluation/#the-examples): Evaluation using LangChain as a framework. Make sure to check the examples. Also read the source code for the example prompts
  - [lang-kit](https://github.com/whylabs/langkit/tree/main) - Library
  - [langsmith](https://docs.smith.langchain.com/)


## LLM In Production
  - [Building LLM applications for production by Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html) ⭐
  - [Generative AI Strategy](https://huyenchip.com/2023/06/07/generative-ai-strategy.html) ⭐
  - [All the Hard Stuff Nobody Talks About when Building Products with LLMs](https://www.honeycomb.io/blog/hard-stuff-nobody-talks-about-llm)
  - Blogs from replit: [here](https://blog.replit.com/llms) & [there](https://blog.replit.com/llm-training)
  - [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers)
  
## Azure OpenAI
  - [Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview)
  - [E2E solutions using Azure OpenAI and various other services](https://github.com/Azure-Samples/openai)
  - [Deployment of an Enterprise Azure OpenAI reference architecture](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/ai/log-monitor-azure-openai)
  - [Bring your own data: Concepts](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/use-your-data)
  - [Azure Promptflow](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow?view=azureml-api-2): Service to evaluate and track LLM Experiments

## Interesting Links
  - [What Is ChatGPT Doing … and Why Does It Work? by Stephen Wolfram](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) 
  - [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)
  - [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
  - [What are embeddings?](https://vickiboykis.com/what_are_embeddings/)

## Transformer Explained
  - [NLP course by hugging face](https://huggingface.co/learn/nlp-course/chapter1/1)
  - [Transformer from Scratch](https://e2eml.school/transformers.html)
  - [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
  - [Some Intuition on Attention and the Transformer](https://eugeneyan.com/writing/attention/#references)
