## SyreaNet: A Physically Guided Underwater Image Enhancement Framework Integrating Synthetic and Real Images

This repository contains the official implementation of the paper:
> **SyreaNet: A Physically Guided Underwater Image Enhancement Framework Integrating Synthetic and Real Images** (accepted by ICRA2023)<br>
> Junjie Wen, Jinqiang Cui, Zhenjun Zhao, Ruixin Yan, Zhi Gao, Lihua Dou, Ben M. Chen <br>
> **Paper Link**: [[arxiv](https://arxiv.org/pdf/2302.08269.pdf)]


## Overview
![overall_arch](./figs/fig-overall_arch.png)
The overall architecture of our method. Specifically, synthetic underwater images are first generated by our proposed physically guided synthesis module (PGSM). Then, various synthetic and real underwater images are fed into the physically guided disentangled network, predicting the clear image, backscattering, transmission and white point. The intra- and inter- Domain Adaptations are done by exchanging the knowledge across attribute domains.

![demo](./figs/fig-demo.png)
Enhancement examples under various underwater conditions. Video can be found at [Youtube](https://www.youtube.com/watch?v=DyOktx7_9JQ).


## Dataset
The synthesis and real-world dataset could be downloaded via:
[BaiduYun](https://pan.baidu.com/s/1iVAR_hSVmLMyrWcjm4HbbA)(Code:90gv)

## Citation
If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{wen2023syreanet,
  title={SyreaNet: A Physically Guided Underwater Image Enhancement Framework Integrating Synthetic and Real Images},
  author={Wen, Junjie and Cui, Jinqiang and Zhao, Zhenjun and Yan, Ruixin and Gao, Zhi and Dou, Lihua and Chen, Ben M},
  journal={arXiv preprint arXiv:2302.08269},
  year={2023}
}
```

### ToDo-List
* [ ] Release the synthesizing code.
* [ ] Release the testing code and model checkpoint.
* [ ] Release the training code.
