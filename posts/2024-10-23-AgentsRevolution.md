---
title:  "AI Agents Revolution"
date:   2024-10-23 07:00:07 +0200
date-modified: last-modified
categories: LLM, Agents
draft: false
description: High level overview of AI Agents and evolutions.
toc: true
from: markdown+emoji


---



# :baby: AI Agents Revolution

![AI Agents](./assets/image-20241023112100724.png)

In the context of Large Language Models (LLMs), AI Agents are autonomous software entities designed to extend the capabilities of LLMs. 

Recently , we have seen several agent frameworks such as [AutoGen](https://microsoft.github.io/autogen/0.2/), [CrewAI](https://www.crewai.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) and examples such as [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) and [BabyAGI](https://github.com/yoheinakajima/babyagi). 

These solutions leverage LLM capabilities and typically consist of the following steps:

- :page_facing_up: **Task decomposition**: break complex tasks into smaller, manageable steps.
- :hammer_and_pick: **Tool integration**: LLM interacting with various tools and APIs to gather data from the environment, such as web scraping tools, calculator and python interpreter.
- :books: **Memory**: used to retain information from past interactions, allowing to provide more context information.
- :currency_exchange: **Autonomous execution**: once plan is established, the agent can execute the steps autonomously.

These steps make AI agents powerful as they extend the functionality of LLMs beyond simple generation tasks towards autonomous task execution. For example, [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning and Acting) is a common methodology applied to AI agents to improve performance by leveraging reasoning and acting capabilities.



![ReAct](./assets/image-20241023090742241.png)

Picture by [Shunyu Yao - LLM Agents - Brief History and Overview](https://rdi.berkeley.edu/llm-agents-mooc/slides/llm_agent_history.pdf)



## :question: What has changed?

So far, AI agent frameworks have needed to communicate with the outside world through tool integration in the form of APIs. However, today Anthropic announced a new approach—**LLMs performing tasks directly on your computer**.

This is possible using **[computer use capability](https://www.anthropic.com/news/3-5-models-and-computer-use)** available in Claude 3.5 sonnet and Claude 3.5 Haiku. 



> Available [today on the API](https://docs.anthropic.com/en/docs/build-with-claude/computer-use?ref=platformer.news), developers can direct Claude to use computers the way people do—by looking at a screen, moving a cursor, clicking buttons, and typing text. Claude 3.5 Sonnet is the first frontier AI model to offer computer use in public beta. At this stage, it is still [experimental](https://www.anthropic.com/news/developing-computer-use?ref=platformer.news) — at times cumbersome and error-prone. We're releasing computer use early for feedback from developers, and expect the capability to improve rapidly over time.



According to [Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use), instead of creating specific tools to help Claude complete individual tasks, we’re **teaching it general computer skills**—allowing it to use a wide range of standard tools and software programs designed for people.

Examples of Anthropic's computer use capability include:

- Creating an entire website on the user's computer and even fixes bugs in the code [[here](https://www.youtube.com/watch?v=vH2f7cjXjKI)].
- Using data from user's computer and online data to fill out forms [[here](https://www.youtube.com/watch?v=ODaHJzOyVCQ&feature=youtu.be)]. 
- Orchestrating a multi-step task by searching the web, using native applications, and creating a plan with the resulting information [[here](https://www.youtube.com/watch?v=jqx18KgIzAE)].

Figure below shows the prompt for the example "Using data from user's computer and online data to fill out forms"  from [youtube video: Computer use for automating operations](https://youtu.be/ODaHJzOyVCQ).

![Anthropic's Computer Use](./assets/image-20241023083751146.png)

Picture by Anthropic's new computer use tool from [Youtube Computer use for automating operations video](https://youtu.be/ODaHJzOyVCQ).



## :telescope: Things to Consider while building AI Agents that works

> This section is based on my notes from the presentation given by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s) on the [SMALLCON](https://eur01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fgo.predibase.com%2Fe3t%2FCtc%2FL0%2B113%2Fd2BmB904%2FVVV0pL45txbtN3W8GY5VDQS-VXkGBG5qDK5DN6-TNZ82-ZVTW5BW0B06lZ3kLN6MyJC5L0v21W4qC8S25HQZPRW431hQ95y2hDTN4pDp4N_RqkrW3bpGmk5PMNdqN7pQ1kbXJFr8W80-y-S1KH_-WW8-X96d16zTjDW2zT-nS8pZsv6W3Cc1203_xn19W58MR6c7yCYkTV6vxMf2H9zDNW5zkmbx1lkw-GN5h3tWGS6sTwN7RR6d6wPzK9MNJdB1KSffvW7q23Gb5xdPHTW27QSw-2RLQ8Gf5-dw5K04&data=05%7C02%7Cmarcel.castro%40km.kongsberg.com%7C73954d3f37b449172bc308dd3140a066%7Ca113bc9e1024489c902e5ac8b5fd41ce%7C0%7C0%7C638720875811932030%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=MeUEEQcmVYUVDzmrWGwmiQNYXGXbjaVs4dce1TwZQT0%3D&reserved=0) conference by Predibase in Dec. 2024.

Important things to consider while building Production-ready AI agents based on steps provided above:

- :page_facing_up: **Task decomposition**: Topics
- :hammer_and_pick: **Tool integration**: Sources, knowledge quality, guardrails
- :books: **Memory**: User context, page context
- :currency_exchange: **Autonomous execution**: actions, intent, latency

In addition, as highlight by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s) customer trust on the agent is important, and common points below need to be considered:

:exclamation: Data privacy and security concerns: e.g. legal

:exclamation: ​lack of confidence in accuracy: "*how this will behave in production?* *I am getting different answer each time*"

:exclamation: ​negative past experience: "*I tried before and it did not work*"

:exclamation: ​ROI and cost concerns: "*how to calculate business value*?"

![image-20250110082320573](./assets/image-20250110082320573.png)

Picture by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s)

As in in software development best practice, the agent development follows similar steps, which are: 

:arrow_right: Ideate :arrow_right:  Setup :arrow_right:  Configure :arrow_right:  Test :arrow_right:  Deploy :arrow_right:  Monitor :repeat:



![image-20250110083443875](./assets/image-20250110083443875.png)

Picture by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s)

### :hammer_and_pick: Test: toolset you need

Testing and evaluation during agent application development  is very important. 

As in all generative AI usecase, the journey normally looks like this:

:arrow_right: Select a pre-trained LLM (e.g. gpt-4o or claude 3.5)

:arrow_right:  Prompt Engineering (improve prompt with few shots) 

:arrow_right:  Prompt Engineering + RAG (longer prompts, slower inference) 

:arrow_right:  Advanced techniques (advanced RAG, finetuning - improve speed, quality and cost) 



Here is an example from [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s) to atomate agent evaluation.

![image-20250110084935905](./assets/image-20250110084935905.png)

Picture by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s)

Below is an example of evaluation while considering a RAG (Retrieval Augmented Generation) pipeline.

![image-20250110085416224](./assets/image-20250110085416224.png)

Picture by [Manjeet Singh, Salesforce](https://www.youtube.com/watch?v=BAL5QzWmt2I&list=PL_lyFNnob30vw-q0_Kttx1DrKOSXoDnwU&t=11s)



## :woman_teacher:  How good are those agents today?



![Agent Evaluations](./assets/image-20241023120005090.png)

Picture by [Philipp Schimid](https://www.linkedin.com/posts/philipp-schmid-a6a2bb196_can-llms-now-control-computers-like-we-do-activity-7254766392921460736-Bx1y?utm_source=share&utm_medium=member_desktop) 



## References:

- [AI Agents: Key Concepts and How They Overcome LLM Limitations by TheNewStack]([AI Agents: Key Concepts and How They Overcome LLM Limitations - The New Stack](https://thenewstack.io/ai-agents-key-concepts-and-how-they-overcome-llm-limitations/))
- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku by Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use) (20241022)
- [Berkley Course: Large Language Model Agents by](https://llmagents-learning.org/f24) [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/)
- [The AI agents have arrived by Casey Newton](https://www.platformer.news/anthropic-ai-agents-computer-use-consequences/?ref=platformer-newsletter)
- [When you give a Claude a mouse by Ethan Mollick](https://www.oneusefulthing.org/p/when-you-give-a-claude-a-mouse?utm_campaign=post&utm_medium=web&ref=platformer.news)
- [LLM Powered Autonomous Agents | Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Agents by Chip Huyen](https://huyenchip.com//2025/01/07/agents.html)

