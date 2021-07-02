# 


# HyMOS (Hyperspherical classification for Multi-source Open-Set domain adaptation)

PyTorch official implementation of  "[Distance-based Hyperspherical Classification for Multi-source Open-Set Domain Adaptation]()" **coming soon**!

![Test Image 1](image.jpeg)

_Vision systems trained in close-world scenarios will inevitably fail when presented with new environmental conditions, new data distributions and novel classes at deployment time. How to move towards open-world learning is along standing research question, but the existing solutions mainly focus on specific aspects of the problem (single domain open-set, multi-domain closed-set), or propose complex strategies which combine multiple losses and manually tuned hyperparameters. In this work we tackle multi-source open-set domain adaptation by introducing HyMOS: a straightforward supervised model that exploits the power of contrastive learning and the properties of its hyperspherical feature space to correctly predict known labels on the target, while rejecting samples belonging to any unknown class. HyMOS  includes a tailored data balancing to enforce cross-source alignment and introduces style transfer among the instance transformations of contrastive learning for source-target adaptation, avoiding the risk of negative transfer. Finally a self-training strategy refines the model without the need for handcrafted thresholds. We validate our method over three challenging datasets and provide an extensive quantitative and qualitative experimental analysis. The obtained results show that HyMOS outperforms several open-set and universal domain adaptation approaches, defining the new state-of-the-art._

**Office-31 (HOS)**

|  | D,A -> W | W,A -> D | W,D -> A | Avg. |
| :---| :---: | :---: | :---: |---: |
| **HyMOS** | 89.6| 89.5| 60.8 | 79.9 |

**Office-Home (HOS)**

|  | Ar,Pr,Cl→Rw | Ar,Pr,Rw→Cl | Cl,Pr,Rw→Ar | Cl,Ar,Rw→Pr | Avg. |
| :---| :---: | :---: | :---: | :---: | ---: |
| **HyMOS** | 71.0 | 64.6 | 62.2 | 71.1 | 67.2 |

**DomainNet (HOS)**

|  | I,P -> S | I,P -> C | Avg. |
| :---| :---: | :---: | ---: 
| **HyMOS** | 57.5| 61.0| 59.3 | 
