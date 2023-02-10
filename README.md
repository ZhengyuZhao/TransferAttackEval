# Evaluating Transfer Adversarial Attacks 

This repository provides the code for our paper: [Towards Good Practices in Evaluating Transfer Adversarial Attacks](https://arxiv.org/abs/2211.09565). Zhengyu Zhao*, Hanwei Zhang*, Renjue Li*, Ronan Sicre, Laurent Amsaleg, Michael Backes.


In this work, we design good practices in evaluating transfer adversarial attacks.
We first introduce a new attack categorization, which enables our systematic and fair analyses of similar attacks in each specific category.
Our analyses lead to new findings that complement or even challenge existing knowledge.
Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet.
We pay particular attention to attack stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics.
Our evaluation reveals new important insights: 1) Transferability is highly contextual, and some white-box defenses may give a false sense of security since they are actually vulnerable to (black-box) transfer attacks; 
2) All transfer attacks are less stealthy, and their stealthiness can vary dramatically under the same $L_{\infty}$ bound.

## Transfer Attack Categorization

### Gradient Stabilization Attacks [[Our implementation]](https://github.com/ZhengyuZhao/TransferAttackEval/tree/main/attacks/gradient_stabilization_attacks.py)
+ [Boosting Adversarial Attacks with Momentum (CVPR 2018)](https://arxiv.org/abs/1710.06081)
+ [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)](https://arxiv.org/abs/1908.06281)
+ [Boosting Adversarial Transferability through Enhanced Momentum (BMVC 2021)](https://arxiv.org/abs/2103.10609)
+ [Improving Adversarial Transferability with Spatial Momentum (arXiv 2022)](https://arxiv.org/abs/2203.13479)

### Input Augmentation Attacks [[Our implementation]](https://github.com/ZhengyuZhao/TransferAttackEval/tree/main/attacks/input_augmentation_attacks.py)
+ [Improving Transferability of Adversarial Examples with Input Diversity (CVPR 2019)](https://arxiv.org/abs/1803.06978)
+ [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks (CVPR 2019)](https://arxiv.org/abs/1904.02884)
+ [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)](https://arxiv.org/abs/1908.06281)
+ [Patch-wise Attack for Fooling Deep Neural Network (ECCV 2020)](https://arxiv.org/abs/2007.06765)
+ [Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting (ECCV 2020)](https://arxiv.org/abs/2112.06011)
+ [Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses (ECCV 2020)](https://arxiv.org/abs/1904.00979)
+ [Enhancing the Transferability of Adversarial Attacks through Variance Tuning (CVPR 2021)](https://arxiv.org/abs/2103.15571)
+ [Admix: Enhancing the Transferability of Adversarial Attacks (ICCV 2021)](https://arxiv.org/abs/2102.00436)
+ [Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input (CVPR 2022)](https://arxiv.org/abs/2203.09123)
+ [Frequency Domain Model Augmentation for Adversarial Attack (ECCV 2022)](https://arxiv.org/abs/2207.05382)
+ [Adaptive Image Transformations for Transfer-based Adversarial Attack (ECCV 2022)](https://arxiv.org/abs/2111.13844)
+ [Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation (NeurIPS 2022)](https://arxiv.org/abs/2210.05968)
+ [Incorporating Locality of Images to Generate Targeted Transferable Adversarial Examples (arXiv 2022)](https://arxiv.org/abs/2209.03716)

### Feature Disruption Attacks [[Our implementation]](https://github.com/ZhengyuZhao/TransferAttackEval/tree/main/attacks/feature_disruption_attacks.py)
+ [Transferable Adversarial Perturbations (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Bruce_Hou_Transferable_Adversarial_Perturbations_ECCV_2018_paper.html)
+ [Task-generalizable Adversarial Attack based on Perceptual Metric (arXiv 2018)](https://arxiv.org/abs/1811.09020)
+ [Feature Space Perturbations Yield More Transferable Adversarial Examples (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Inkawhich_Feature_Space_Perturbations_Yield_More_Transferable_Adversarial_Examples_CVPR_2019_paper.html)
+ [FDA: Feature Disruptive Attack (ICCV 2019)](https://arxiv.org/abs/1909.04385)
+ [Enhancing Adversarial Example Transferability with an Intermediate Level Attack (ICCV 2019)](https://arxiv.org/abs/1907.10823)
+ [Transferable Perturbations of Deep Feature Distributions (ICLR 2020)](https://arxiv.org/abs/2004.12519)
+ [Boosting the Transferability of Adversarial Samples via Attention (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_Boosting_the_Transferability_of_Adversarial_Samples_via_Attention_CVPR_2020_paper.html)
+ [Yet Another Intermediate-Level Attack (ECCV 2020)](https://arxiv.org/abs/2008.08847)
+ [Perturbing Across the Feature Hierarchy to Improve Standard and Strict Blackbox Attack Transferability (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/hash/eefc7bfe8fd6e2c8c01aa6ca7b1aab1a-Abstract.html)
+ [Feature Importance-aware Transferable Adversarial Attacks (ICCV 2021)](https://arxiv.org/abs/2107.14185)
+ [Improving Adversarial Transferability via Neuron Attribution-Based Attacks (CVPR 2022)](https://arxiv.org/abs/2204.00008)
+ [An Intermediate-level Attack Framework on The Basis of Linear Regression (TPAMI 2022)](https://arxiv.org/abs/2203.10723)

### Surrogate Refinement Attacks [[Our implementation]](https://github.com/ZhengyuZhao/TransferAttackEval/tree/main/attacks/surrogate_refinement_attacks.py)
+ [Learning Transferable Adversarial Examples via Ghost Networks (AAAI 2020)](https://arxiv.org/abs/1812.03413)
+ [Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets (ICLR 2020)](https://arxiv.org/abs/2002.05990)
+ [Backpropagating Linearly Improves Transferability of Adversarial Examples (NeurIPS 2020)](https://arxiv.org/abs/2012.03528)
+ [Backpropagating Smoothly Improves Transferability of Adversarial Examples (CVPRw 2021)](https://aisecure-workshop.github.io/amlcvpr2021/cr/31.pdf)
+ [A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks (NeurIPS 2021)](https://arxiv.org/abs/2106.02105)
+ [Early Stop And Adversarial Training Yield Better surrogate Model: Very Non-Robust Features Harm Adversarial Transferability (OpenReview 2021)](https://openreview.net/forum?id=ECC7T-torK)
+ [Training Meta-Surrogate Model for Transferable Adversarial Attack (arXiv 2021)](https://arxiv.org/abs/2109.01983)
+ [Rethinking Adversarial Transferability from a Data Distribution Perspective (ICLR 2022)](https://openreview.net/forum?id=gVRhIEajG1k)
+ [Boosting the Adversarial Transferability of Surrogate Model with Dark Knowledge (arXiv 2022)](https://arxiv.org/abs/2206.08316)

### Generative Modeling Attacks
+ [Generative Adversarial Perturbations (CVPR 2018)](https://arxiv.org/abs/1712.02328) [[code]](https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations)
+ [Cross-Domain Transferability of Adversarial Perturbations (NeurIPS 2019)](https://arxiv.org/abs/1905.11736) [[code]](https://github.com/Muzammal-Naseer/Cross-Domain-Perturbations)
+ [On Generating Transferable Targeted Perturbations (ICCV 2021)](https://arxiv.org/abs/2103.14641) [[code]](https://github.com/Muzammal-Naseer/TTP)
+ [Learning Transferable Adversarial Perturbations (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html) [[code]](https://github.com/krishnakanthnakka/Transferable_Perturbations)
+ [Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains (ICLR 2022)](https://arxiv.org/abs/2201.11528) [[code]](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack)
+ [Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks (ECCV 2022)](https://arxiv.org/abs/2107.01809) [[code]](https://github.com/ShawnXYang/C-GSP)

## Other General Analyses of Transfer Adversarial Attacks
+ [Delving into Transferable Adversarial Examples and Black-box Attacks (ICLR 2017)](https://arxiv.org/abs/1611.02770)
+ [The Space of Transferable Adversarial Examples (arXiv 2017)](https://arxiv.org/abs/1704.03453)
+ [Why Do Adversarial Attacks Transfer? Explaining Transferability of Evasion and Poisoning Attacks (USENIX Security 2019)](https://arxiv.org/abs/1809.02861)
+ [Towards Transferable Targeted Attack (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Towards_Transferable_Targeted_Attack_CVPR_2020_paper.html)
+ [Understanding and Enhancing the Transferability of Adversarial Examples (ACML 2020)](https://arxiv.org/abs/1802.09707)
+ [Selection of Source Images Heavily Influences the Effectiveness of Adversarial Attacks (BMVC 2021)](https://arxiv.org/abs/2106.07141)
+ [On Success and Simplicity: A Second Look at Transferable Targeted Attacks (NeurIPS 2021)](https://arxiv.org/abs/2012.11207)
+ [Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes (NeurIPSw 2021)](https://arxiv.org/abs/2111.11056)
+ [Investigating Top-k White-Box and Transferable Black-Box Attack (CVPR 2022)](https://arxiv.org/abs/2204.00089)
+ [Transfer Attacks Revisited: A Large-Scale Empirical Study in Real Computer Vision Settings (IEEE S&P 2022)](https://arxiv.org/abs/2204.04063)
