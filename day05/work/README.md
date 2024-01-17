# LMDeploy

- [300 字小故事](#TurboMind+Python-生成-300-字小故事)

- [Internlm-chat-7b KV Cache 量化](#Internlm-chat-7b-KV-Cache-量化)

### TurboMind+Python 生成 300 字小故事

```python
from lmdeploy import turbomind as tm

# load model
model_path = "/root/share/temp/model_repos/internlm-chat-7b/"
tm_model = tm.TurboMind.from_pretrained(model_path, model_name='internlm-chat-7b')
generator = tm_model.create_instance()

# process query
query = " 写一段 300 字的小故事"
prompt = tm_model.model.get_prompt(query)
input_ids = tm_model.tokenizer.encode(prompt)

# inference
for outputs in generator.stream_infer(
        session_id=0,
        input_ids=[input_ids]):
    res, tokens = outputs[0]

response = tm_model.tokenizer.decode(res.tolist())
print(response)
```

![image-20240112203258931](README.assets/image-20240112203258931.png)



### Internlm-chat-7b KV Cache 量化

- **计算 minmax**

  ```bash
  # 计算 minmax
  lmdeploy lite calibrate \
    --model  /root/share/temp/model_repos/internlm-chat-7b/ \
    --calib_dataset "c4" \
    --calib_samples 128 \
    --calib_seqlen 2048 \
    --work_dir ./quant_output
  ```

  选择 128 条输入样本，每条样本长度为 2048，数据集选择 C4

  > 这一步由于默认需要从 Huggingface 下载数据集，国内经常不成功。所以我们导出了需要的数据，大家需要对读取数据集的代码文件做一下替换。共包括两步：
  >
  > 第一步：复制 `calib_dataloader.py` 到安装目录替换该文件：`cp /root/share/temp/datasets/c4/calib_dataloader.py  /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/lite/utils/`
  >
  > 第二步：将用到的数据集（c4）复制到下面的目录：`cp -r /root/share/temp/datasets/c4/ /root/.cache/huggingface/datasets/`

  