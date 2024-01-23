# Euler

### 微调

- 命令行

  ```bash
  xtuner train llama2_7b_chat_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2
  ```

- 训练结果

  <img src="README.assets/image-20240123174531621.png" alt="image-20240123174531621" style="zoom:67%;" />

- 转换成`hf`格式

  ```bash
  mkdir hf_llama
  export MKL_SERVICE_FORCE_INTEL=1
  xtuner convert pth_to_hf ./llama2_7b_chat_qlora_oasst1_e3_copy.py ./work_dirs/llama2_7b_chat_qlora_oasst1_e3_copy/epoch_2.pth/ ./hf_llama/
  ```

  ![image-20240123174933672](README.assets/image-20240123174933672.png)
  
  ![image-20240123174955819](README.assets/image-20240123174955819.png)
  
  LoRA 模型文件 = Adapter
  
- 整合模型

  ```bash
  xtuner convert merge model/Llama-2-7b-chat-hf/ ./hf_llama/ ./merged_llama/
  ```

  ![image-20240123175324966](README.assets/image-20240123175324966.png)
  
- 使用模型

  ![image-20240115141114357](README.assets/image-20240115141114357.png)

  ```bash
  export MKL_SERVICE_FORCE_INTEL=1
  xtuner chat ./merged_llama/ --prompt-template llama2_chat --system 'If you are a math teacher, please answer a math question, and please give the process and answer to the problem based on your thinking. Answer in Chinese'
  ```

  ![image-20240123181502528](README.assets/image-20240123181502528.png)



### 部署

- 转换模型

  ```bash
  lmdeploy convert llama ./merged_llama/
  ```

- 量化

  ```bash
  lmdeploy lite calibrate \
    --model ./merged_llama/ \
    --calib_dataset "ptb" \
    --calib_samples 128 \
    --calib_seqlen 2048 \
    --work_dir ./quant_output
  ```

  <img src="README.assets/image-20240123182503954.png" alt="image-20240123182503954" style="zoom:67%;" />

  ```bash
  lmdeploy lite kv_qparams \
    --work_dir ./quant_output  \
    --turbomind_dir workspace/triton_models/weights/ \
    --kv_sym False \
    --num_tp 1
  ```

  ```bash
  lmdeploy chat turbomind ./workspace
  ```

  ![image-20240123182729293](README.assets/image-20240123182729293.png)