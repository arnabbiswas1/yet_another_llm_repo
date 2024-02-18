# Getting Started with GenAI : Manager’s Edition

[Last Update : Feb' 2024]

Few of my friends reached out to me asking for resources to get started with GenAI. Most of them are in senior management/tech roles, trying to identify use cases where GenAI can bring value. With that goal, they are willing to build an intuitive understanding. Here are few pointers which "I think" would be useful for them (Links in comment):

1. [Generative AI for Everyone by Andrew NG](https://www.deeplearning.ai/courses/generative-ai-for-everyone/) : A perfect course for business leaders. Explains the concepts, walks through basic uses cases, discusses how to identify opportunities, build teams etc. It's a paid course (3 - 6 hours duration). Audit the course before buying.

2. [Catching up on the weird world of LLMs by Simon Willson](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/): This 40 min talk will familiarize you with all the buzz words, help to understand about Large Language Model (LLM) from end user perspective. Slightly outdated (Aug' 2023), but one of my favorites.

3. [Intro to LLM by Andrej Karpathy](https://youtu.be/zjkBMFhNj_g?si=XDx6Su-hgu_k7eZV): An hour talk which is little more technical. It will help to build an intuitive understanding about how LLM works. In the 2nd half, he talks about security vulnerabilities of LLMs in production.

At this point, I will suggest to start playing with services like ChatGPT, Gemini, Bing Chat (now Copilot). If possible, try out the paid version. Killer features are often behind pay wall (Gemini Advanced has 2 months of free trial, ChatGPT Plus costs $20/month. ChatGPT Plus subscription doesn't provide API access). Try whatever you can think of: text, audio, images, code, web search, csv files, assistants (GPTs). But, keep in mind, these are not LLMs; these are services using LLMs in the backend together with many other tools. For example, ChatGPT-4 Plus uses GPT-4 as LLM, Dall-E as diffusion model (for image creation), a web browser plugin (to fetch data from the internet) and a code interpreter plugin (to answer questions by writing and executing code). Same is true for Bing Chat or Gemini. LLMs are good at language generation and understanding. They are reasoning engines. The magic happens only when LLMs are attached to other tools to perform various tasks. That is why, next, I would suggest you to taste the "raw" capabilities of LLMs (your developers will finally use those over APIs).

Here comes [Chatbot Arena by LMSYS](https://chat.lmsys.org/). You can interact with models like GPT4, Gemini Pro or Mistral Medium etc. LMSYS is a research organization from UC Berkley. So, please provide your feedback after every chat conversation. They rank the models based on our feedback.

What about the use cases? langchain (Popular framework to build application using LLMs) has a [page](https://python.langchain.com/docs/use_cases) dedicated for this. It is built for developers using the framework, but it will help to quickly become familiar with various applications people are building with LLMs. If you are aware of something better, please let me know.

Finally, a note of caution. GenAI is at the top of the hype cycle now. Take everything with a pinch of salt.