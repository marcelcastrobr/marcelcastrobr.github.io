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



In the context of Large Language Models (LLMs), AI Agents are autonomous software entities designed to extend the capabilities of LLMs. 

Lately, we have seen several agents frameworks such as AutoGen, CrewAI and LangGraph and examples such as AutoGPT and BabyAGI. 

 So far those solutions leverage LLM capabilities and normally are composed by the following steps:

- :page_facing_up: **Task decomposition**: break complex tasks into smaller, manageable steps
- :hammer_and_pick: **Tool integration**: LLM interacting with various tools and APIs to gather data from the environment, such as web scraping tools, calculator and python interpreter
- :books: **Memory**: used to retain information from past interactions, allowing to provide more context information.
- :currency_exchange: **Autonomous execution**: once plan is established, the agent can execute the steps autonomously.

Those steps make AI agents powerful as it extend the functionality of LLMs beyond simple generation tasks towards autonomous task execution. For example, ReAct (Reasoning and Acting) is a common methodology applied to AI agents to improve performance by leveraging reasoning and acting capabilities. 



![image-20241023090742241](./assets/image-20241023090742241.png)

Picture by [Shunyu Yao - LLM Agents - Brief History and Overview](https://rdi.berkeley.edu/llm-agents-mooc/slides/llm_agent_history.pdf)



## :question: What has changed?

So far, the AI agent frameworks needed to talk to the outside world through **tool integration** in the form of APIs. But today Anthropic announced a new way - LLM doing tasks on your computer for you.  

This is possible using **[computer use capability](https://www.anthropic.com/news/3-5-models-and-computer-use)** available in Claude 3.5 sonnet and Claude 3.5 Haiku. 



> Available [today on the API](https://docs.anthropic.com/en/docs/build-with-claude/computer-use?ref=platformer.news), developers can direct Claude to use computers the way people do—by looking at a screen, moving a cursor, clicking buttons, and typing text. Claude 3.5 Sonnet is the first frontier AI model to offer computer use in public beta. At this stage, it is still [experimental](https://www.anthropic.com/news/developing-computer-use?ref=platformer.news) — at times cumbersome and error-prone. We're releasing computer use early for feedback from developers, and expect the capability to improve rapidly over time.



According to [Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use), *instead of making specific tools to help Claude complete individual tasks, we're **teaching it general computer skills**—allowing it to use a wide range of standard tools and software programs designed for people.*

Examples of computer use capability by Anthropic are for example:

- creation of an entire website on the user's computer and even fixes bugs in the code [[here](https://www.youtube.com/watch?v=vH2f7cjXjKI)].
- Use data from user's computer and online data to fill out forms [[here](https://www.youtube.com/watch?v=ODaHJzOyVCQ&feature=youtu.be)]. 
- Be able to orchestrate a multi-step task by searching the web, using native applications, and creating a plan with the resulting information [[here](https://www.youtube.com/watch?v=jqx18KgIzAE)].



![image-20241023083751146](./assets/image-20241023083751146.png)

Picture by Anthropic's new computer use tool from [Youtube Computer use for automating operations video](https://youtu.be/ODaHJzOyVCQ).





## References:

- [AI Agents: Key Concepts and How They Overcome LLM Limitations by TheNewStack]([AI Agents: Key Concepts and How They Overcome LLM Limitations - The New Stack](https://thenewstack.io/ai-agents-key-concepts-and-how-they-overcome-llm-limitations/))
- [Introducing computer use, a new Claude 3.5 Sonnet, and Claude 3.5 Haiku by Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use) (20241022)
- [Berkley Course: Large Language Model Agents by](https://llmagents-learning.org/f24) [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/)
- [The AI agents have arrived by Casey Newton](https://www.platformer.news/anthropic-ai-agents-computer-use-consequences/?ref=platformer-newsletter)
- [When you give a Claude a mouse by Ethan Mollick](https://www.oneusefulthing.org/p/when-you-give-a-claude-a-mouse?utm_campaign=post&utm_medium=web&ref=platformer.news)
