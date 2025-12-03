<h2 align="center"> <a href="https://arxiv.org/abs/2504.14096">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></h2>
<h3 align="center">EMNLP 2025 (Main Conference)</h3>
<div align="center">


<br>


<a href='https://arxiv.org/abs/2504.14096'><img src='https://img.shields.io/badge/arXiv-2504.14096-b31b1b.svg'></a> &nbsp;
 <a href='https://people-robots.github.io/VideoPASTA/'><img src='https://img.shields.io/badge/Project-Website-blue'></a>&nbsp;
 <a href='https://huggingface.co/yogkul2000/VideoPASTA'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a> 
 <a href='https://huggingface.co/datasets/yogkul2000/VideoPASTA-7k'><img src='https://img.shields.io/badge/huggingface-datasets-green'></a> 

 
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
- [x] Release VideoPASTA fine-tuned model weights (e.g., Qwen2.5-VL + VideoPASTA).
- [x] Release Inference Code.
- [ ] Release VideoPASTA Preference Dataset.

## 📦 Install

### Environment Setup

```bash
conda create -n videopasta python=3.10
conda activate videopasta

# Install PyTorch with CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# Install flash attention (if facing issues, use the command below)
pip install flash-attn==2.7.4.post1

# If flash-attn installation fails, try:
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install transformers==4.54.1

# Install other dependencies
pip install decord opencv-python pillow numpy
pip install qwen-vl-utils==0.0.10

# Install vLLM with a specific CUDA version
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=126
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

## 📝 Data

### Training Data (VideoPASTA-7K)
* Our framework uses **VideoPASTA-7K**, a dataset comprising 7,020 preference pairs $(V, q, r^+, r^-)$.
* Each pair contains an input video ($V$), a query ($q$), a preferred response ($r^+$) aligned with the video, and an adversarial response ($r^-$) designed to introduce spatial, temporal, or cross-frame misalignment.
* The data was generated using videos sampled from ActivityNet and filtered using Qwen2.5-32B.

### Evaluation Data
We evaluated VideoPASTA on the following standard benchmarks:
* **General Video Understanding:** TempCompass, Perception-Test, NeXTQA, MVBench.
* **Long-Form Video Understanding:** LongVideoBench, MLVU, VideoMME.

#### Individual Benchmark Evaluation

```bash
python3 -m lmms_eval \
    --model vllm \
    --model_args model="$MODEL_PATH",tensor_parallel_size="$INFERENCE_TP_SIZE",gpu_memory_utilization=0.9 \
    --tasks longvideobench_val_v \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_PATH"
```

## 📝 Citation
If you find VideoPASTA useful for your research, please cite our paper:
```bib
@article{kulkarni2025videopasta,
  title={{VideoPASTA}: 7K Preference Pairs That Matter for Video-LLM Alignment},
  author={Kulkarni, Yogesh and Fazli, Pooyan},
  journal={arXiv preprint arXiv:2504.14096},
  year={2025}
}
```

## 📪 Contact
For questions about the paper, please contact Yogesh Kulkarni at `ykulka10@asu.edu`. You can also open an issue in this GitHub repository for bugs or specific questions related to the code.

