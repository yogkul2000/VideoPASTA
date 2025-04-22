<h2 align="center"> <a href="https://arxiv.org/abs/2504.14096">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></h2>

<div align="center">


<br>


<a href='https://arxiv.org/abs/2504.14096'><img src='https://img.shields.io/badge/arXiv-2504.14096-b31b1b.svg'></a> &nbsp;
 <a href='https://people-robots.github.io/VideoPASTA/'><img src='https://img.shields.io/badge/Project-Website-blue'></a>&nbsp;
 <a href='#'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20VideoPASTA--7K-Dataset-blue'></a>&nbsp;
 <a href='#'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 

 
 </div>

## Abstract
Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully generated adversarial examples that deliberately violate spatial, temporal, or cross-frame relations. By applying Direct Preference Optimization to just 7,020 preference pairs, VideoPASTA learns robust representations that capture fine-grained spatial relationships and long-range temporal dynamics. Experiments on standard video benchmarks show significant relative performance gains over baseline models. These results demonstrate that targeted alignment effectively addresses core video-language challenges without massive pretraining or architectural modifications. Notably, VideoPASTA achieves these improvements without human annotation or captioning, relying on just 32-frame sampling.

<table class="center">
    <tr>
    <td><img src="assets/pipeline.png" alt="VideoPASTA Overview Diagram"></td>
    </tr>
    <tr>
    <td align="center"><em>Overview of the VideoPASTA framework.</em></td>
    </tr>
</table>

## 🧰 TODO
- [x] Release Paper.
- [ ] Release VideoPASTA-7K Dataset.
- [ ] Release VideoPASTA fine-tuned model weights.
- [ ] Release Training Code.
- [ ] Release Inference Code.
- [ ] Add detailed evaluation scripts.

## 📝 Data

### Training Data (VideoPASTA-7K)
* Our framework uses **VideoPASTA-7K**, a dataset comprising 7,020 preference pairs $(V, q, r^+, r^-)$.
* Each pair contains an input video ($V$), a query ($q$), a preferred response ($r^+$) aligned with the video, and an adversarial response ($r^-$) designed to introduce spatial, temporal, or cross-frame misalignment.
* The data was generated using videos sampled from ActivityNet and filtered using Qwen2.5-32B.
* The dataset will be released soon. `[Link to Dataset when available]`

### Evaluation Data
We evaluated VideoPASTA on the following standard benchmarks:
* **General Video Understanding:** TempCompass, Perception-Test, NeXTQA, MVBench, Vinoground.
* **Long-Form Video Understanding:** LongVideoBench, MLVU, VideoMME.





## 📝 Citation
If you find VideoPASTA useful for your research, please cite our paper:
```bib
@article{kulkarni2025videopasta,
  title={{VideoPASTA}: 7K Preference Pairs That Matter for {Video-LLM} Alignment},
  author={Kulkarni, Yogesh and Fazli, Pooyan},
  journal={arXiv preprint arXiv:2504.14096},
  year={2025},
  url={[https://arxiv.org/abs/2504.14096](https://arxiv.org/abs/2504.14096)}
}
```

## 📪 Contact
For questions about the paper, please contact Yogesh Kulkarni at `ykulka10@asu.edu`. You can also open an issue in this GitHub repository for bugs or specific questions related to the code.

