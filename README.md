## Federated PPO for Cooperative Policy Optimization

---

### Motivation and Background  
Reinforcement Learning with Human Feedback (RLHF) has become the standard for fine-tuning large language models (LLMs), leveraging human guidance to enhance performance. However, as highlighted in [OpenFedLLM (Ye et al., 2024)](https://arxiv.org/abs/2402.06954):  

> "Trained on massive publicly available data, large language models (LLMs) have demonstrated tremendous success across various fields. While more data contributes to better performance, a disconcerting reality is that high-quality public data will be exhausted in a few years. To address this, collaborative and privacy-preserving LLM training on underutilized distributed private data via federated learning (FL) offers a promising solution."

The key challenge is enabling isolated LLMs to share knowledge securely and effectively, improving performance while maintaining privacy. Our goal is to integrate **Federated Learning (FL)** and **RLHF** to enable decentralized, privacy-preserving cooperation among LLMs.

---

### Related Work  

- **Federated Learning (FL)**. Classical approaches like FedAvg ([McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) and policy gradient sharing ([Geyer et al., 2017](https://arxiv.org/abs/1712.07557)) laid the groundwork for distributed training. However, their computational demands and potential privacy risks limit their applicability to LLM pipelines.  

- **Reinforcement Learning with Human Feedback** ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741); [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)) improves model alignment with human preferences using techniques like PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) and DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)). Despite its effectiveness, RLHF remains largely centralized, limiting scalability and privacy.  

- **Federated LLM Fine-Tuning**. Recent works ([Ye et al., 2024](https://arxiv.org/abs/2402.06954); [Sun et al., 2024](https://arxiv.org/abs/2403.12313); [Zhang et al., 2023](https://arxiv.org/abs/2305.05644)) integrate FL with Parameter-Efficient Fine-Tuning (PEFT) for LLMs. While advancing privacy-preserving training, these approaches do not incorporate RLHF directly.  

- **Federated RLHF** has only been explored by [Feijie Wu et al. 2024](https://arxiv.org/abs/2407.03038), [2025](https://openreview.net/forum?id=mqNKiEB6pd), who propose a DPO-based method. Other RLHF approaches, like PPO, remain unexplored.  

**Gap in Literature**. No existing work examines RLHF methods such as PPO within FL frameworks. Addressing this gap could enable more scalable, privacy-preserving, and effective LLM fine-tuning.

---

### Our Contribution

Our primary contribution is the development of the Federated PPO algorithm, which enables communication between agents through a generalized KL-penalty. Traditionally used as a trust-region soft constraint for stable training, we extend the KL-penalty to act as an optimization trajectory attractor. This novel use aligns individual policies with the learning directions of other agents, enabling private yet effective information exchange and enhancing collaborative training.

We implemented the algorithm pipeline using the Hugging Face framework and conducted initial toy experiments. These results demonstrate promising performance improvements for collaboratively trained models compared to isolated training.

---

### Reproducibility

To run experiments you would need to install our `ppotune` package into your environment
```
git clone https://github.com/RLHF-And-Friends/TunePPO ~/destination
pip install ~/destination
```
Download Mistral models (choose prefix of your preference)
```
# 1. Download the reward model
tune download weqweasdas/RM-Mistral-7B --output-dir prefix/models/RM-Mistral-7B/
# 2a. Get base Mistral-7B model
tune download mistral-community/Mistral-7B-v0.2 --output-dir prefix/models/Mistral-7B-Base/
# 2b. Or get Mistral-7B instruct model
tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir prefix/models/Mistral-7B-Instruct/
```
Take a look at recipe config `~/destination/receipts/ppo/configs/mistral_7b.yaml` and set prefix variable there. This one is needed for the recipe to know where to load and save models, logs etc. Change other parameters to your preference.

Run the recipe for N processes (one per agent) starting from 1. Single process run would be a regular PPO. Non-quantized setup would take something like single 40Gb A100 per process.
```
tune run ~/destination/receipts/ppo/ppo.py --nproc_per_node N --config ~/destination/receipts/ppo/configs/mistral_7b.yaml
```
This is the very first version so some workaround would be inevitable. Feel free to check out torchtune docs like ones about [recipes](https://pytorch.org/torchtune/stable/deep_dives/recipe_deepdive.html#recipe-deepdive) and [configs](https://pytorch.org/torchtune/stable/deep_dives/configs.html) to figure out what's going on here.
