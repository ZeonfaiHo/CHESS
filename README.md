# Exporting CHESS Models: A Simple Guide 

using Llama-3 8B as an example

## Prerequisites

- Download the Llama-3 8B weights and place them in `./base_model/Meta-Llama-3-8B-hf`
- Download the C4 dataset and place it in `./dataset/c4`

## Steps

1. **Export a customized Llama-3 model for activation statistics and threshold computation**
    - Run the Jupyter notebook: `./llama_3_8b/statistics/export_custom_llama.ipynb`

2. **Extract a subset from the C4 dataset and tokenize it**
    - Run the Jupyter notebook: `./llama_3_8b/preprocess.ipynb`

3. **Compute necessary thresholds for CATS, CHESS w/o, and CHESS w/ models**
    - Run the Jupyter notebook: `./llama_3_8b/statics.ipynb`
    - You can adjust the sparsity level by modifying the `sparsity_level` variable in the script

4. **Export CATS and CHESS models with thresholds**
    - Run the Jupyter notebook: `./llama_3_8b/CHESS/export_custom_llama.ipynb`

5. **Run CATS and CHESS models**
    - Use `AutoModelForCausalLM.from_pretrained(model_directory)` method to load our customized models
    - Activation sparsification configuration can be adjusted in config.json
        - `activation_sparsity_type` can be set to
            - `None`: the original base model
            - `CATS`: the base model with FFN activation sparsification as described in CATS [link](https://arxiv.org/abs/2404.08763)
            - `CATS+` (CHESS w/o): the base model with channel-wise thresholding for FFN, as described in our paper
            - `CATS++` (CHESS W/): the base model with channel-wise thresholding for FFN and selective sparsification for attention, as described in our paper
        - Our custom sparse kernels have shown good performance on the Core i9-12900K CPU, but performance on other CPUs may still need optimization and could encounter compatibility issues. We are actively optimizing our custom kernels to achieve the best efficiency on a broad range of CPUs. We will release these kernels once they become available. In the meantime, please disable `use_spvmm_cpu` and `use_vmmsp_cpu` for testing.