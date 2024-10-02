---
title:  "Thinking Tokens"
date:   2024-10-01 07:00:07 +0200
categories: LLM, transformers, Tokens
draft: false
description: Concept of thinking tokens to improve model performance while reasoning.
toc: true


---



# Thinking Tokens



**Thinking tokens** concept (also known as reasoning tokens) enables more intelligence to large models during inference. Until now, the rule to get more intelligent models was only possible through pre-training large model following the "scaling laws", i.e. adding more training data and computing to pretrain large models.

Now with the concept of "thinking tokens" you can achieve more inteligence with the introduction of an internal model reasoning while doing the next token prediction.  

> <|startofthought|> and <|endofthought|>

The idea of thinking tokens has been introduced by some authors such as [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) and latest [o1 model](https://platform.openai.com/docs/guides/reasoning) from OpenAI. Thinking tokens are named reasoning tokens by OpenAI.

The basic concept is to generate "thinking tokens" at inference time to help model to predict next token. A key challenge is to efficiently generate rationales at each token position in the input sequence. However, as pointed out by simply creating a separate forward pass for each token would be computationally intractable for longer sentences.

![image-20241002100759413](./assets/image-20241002100759413.png)

Picture:  [Quiet-STaR](https://arxiv.org/abs/2403.09629)

According to authors, this is done at the inference pass of a language model, when it produces the probability distribution over the next tokens for all input tokens.  The solution in Quiet-STaR implements it by *caching each forward pass and concatenating a diagonal attention mask to the previous attention mask. Thus each generated token attends to all of the tokens that were used to generate it, as well as itself.* But it does not consider the token on the other "counterfactual" paths.



![image-20241002101133078](./assets/image-20241002101133078.png)

> Interestingly, not all tokens requires equal amount of thought . 

Interestingly, not all tokens requires equal amount of thought .  Thus the thinking token technique does not benefit all tokens equally. For example the sentence "**the person is run-**", the "**ing**" is most probably the token with highest probability and there the additional thinking is unlike to improve a well-trained prediction model.

Thus complex reasoning task such as GSM8K are the ones that would benefit more from the thinking token technique.

![image-20241002101210020](./assets/image-20241002101210020.png)

Results:

> Amount of thinking tokens increase the accuracy of the models.

As show in figure below, more  thinking tokens improve the GSM8K accuracy as the training steps icreases. 

![image-20241002100915950](./assets/image-20241002100915950.png)



##  Chain of Thought (CoT) Differences

Some differences highlighted by Quiet-STaR authors ([here](https://community.openai.com/t/papers-quiet-star-language-models-can-teach-themselves-to-think-before-speaking/686158/3)) while comparing thinking tokens to CoT are:

- Different from CoT, model is trained using RL (reinforcement learning) to generate more useful thoughts.
- Rewards model used to generate inner monologues that helps to predict text instead of answers to specific questions - less domain specific. 

As pointed by OpenAI [here](https://platform.openai.com/docs/guides/reasoning/advice-on-prompting)  CoT might undermine "thinking tokens". Thus a best practice are:

- **Avoid chain-of-thought prompts:** Since these models perform reasoning internally, prompting them to "think step by step" or "explain your reasoning" is unnecessary.
- **Limit additional context in retrieval-augmented generation (RAG):** When providing additional context or documents, include only the most relevant information to prevent the model from overcomplicating its response.

## References:

[Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)

[Reasoning Models by OpenAI](https://platform.openai.com/docs/guides/reasoning)