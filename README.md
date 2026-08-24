# LLM Infra Audit — Ornith-1.0 Dual-Node Cluster

稽核日期：2026-08-24
稽核方式：SSH 遠端探查（`ssh -i <ephemeral key>`），未存取任何模型輸出內容，未呼叫推論 API。

## TL;DR

兩台 **NVIDIA DGX Spark（GB10 Grace Blackwell Superchip）**，透過 4×200Gb/s RoCE 網路互聯，
以 **vLLM Tensor-Parallel (TP=2, 2 nodes)** 方式共同載入並服務一個自建的
**397B 參數、512-expert MoE、多模態（Qwen3.5-MoE 架構）模型「Ornith-1.0」**，
INT4 (AutoRound W4A16) 量化後磁碟佔用 196GB，經 OpenAI 相容 API（帶金鑰驗證）對內網提供服務。

## 節點總覽

| 項目 | Host A | Host B |
|---|---|---|
| SSH | `vghks7@172.17.161.28` | `vghks15@172.17.161.30` |
| Hostname | `edgexpert-542a` | `edgexpert-0d4a` |
| OS | Ubuntu 24.04.4 LTS (aarch64) | Ubuntu 24.04.4 LTS (aarch64) |
| Kernel | 6.17.0-1026-nvidia | 6.17.0-1026-nvidia |
| GPU | 1× NVIDIA GB10 | 1× NVIDIA GB10 |
| Driver / CUDA | 580.159.03 / CUDA 13.0 | 580.159.03 / CUDA 13.0 |
| 節點間私網 IP | `10.0.0.1/24` | `10.0.0.x/24` |
| 節點互聯 NIC | 4× RoCE (`rocep1s0f0/f1`, `roceP2p1s0f0/f1`) @ 200 Gb/s | 同左 |
| GPU↔NIC 拓撲 | NODE（同 NUMA node，經 PCIe host bridge） | 同左 |

這兩台實質上是 NVIDIA DGX Spark 的「雙機串接」部署方式：單機 GPU 記憶體不足以放下 397B 模型，
用高速 RoCE 網路把兩台的 GPU 記憶體虛擬合併，透過 tensor parallel 切分權重運算。

## 推論服務堆疊

- **容器化**：Docker，image `vllm-node:latest`（19.2GB），container 名稱 `vllm_node`，`network_mode: host`
  - container 本身 entrypoint 是 `sleep infinity`，vLLM 進程是另外用 `docker exec` 手動啟動並常駐（非 container CMD 直接跑，因此 `docker logs` 抓不到 vLLM 自己的 log，需另外找 log 檔或 journal）
- **推論引擎**：vLLM `0.23.1rc1.dev1043+ga4b4b5787.d20260711`（自建 dev build）
- **相依套件版本**：
  - `torch 2.11.0+cu130` / `torchvision 0.26.0+cu130` / `torchaudio 2.11.0+cu130`
  - `transformers 5.13.1`
  - `flashinfer-python 0.6.15`（+ `flashinfer-cubin` / `flashinfer-jit-cache`）
- **啟動指令**（兩台幾乎相同，經整理去除機敏資訊）：

  ```
  vllm serve /models/Ornith-1.0-397B-W4A16-AutoRound \
    --served-model-name ornith \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization-gb 103 \
    --kv-cache-memory-bytes 4294967296 \
    --port 8000 --host 0.0.0.0 \
    --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --max-num-batched-tokens 4176 \
    --trust-remote-code \
    --load-format instanttensor \
    --api-key sk-ornith-************************** \
    --chat-template /models/chat_template_fixed.jinja \
    -tp 2 --nnodes 2 --node-rank <0|1> \
    --master-addr 10.0.0.1 --master-port 29501
  ```

- **對外埠**：`0.0.0.0:8000`，OpenAI 相容 API（`/v1/models` 等），**有 API key 驗證**
  （`curl localhost:8000/v1/models` 未帶 key 回傳 `{"error":"Unauthorized"}`，確認保護有效）
- **Docker mounts**（host bind mounts，非 volume）：
  - `/var/tmp/ornith-models` → `/models`
  - `~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.tilelang`, `~/.triton` → 對應容器內快取路徑

## 模型架構：Ornith-1.0-397B-W4A16-AutoRound

- **模型類別**：`Qwen3_5MoeForConditionalGeneration`（Qwen3.5 系列 MoE 架構的自訓練/微調衍生模型，非官方 Qwen 命名）
- **多模態**：含 Vision Encoder（ViT，27 層，hidden 1152，patch16，temporal_patch2，支援圖片輸入）
- **文字模型（`text_config`）**：
  - `hidden_size` 4096，`num_hidden_layers` 60
  - **混合注意力機制**：60 層中每 4 層一次 `full_attention`，其餘 45 層為 `linear_attention`（類 Mamba/線性注意力的 hybrid 架構）
  - MoE：`num_experts` **512**，`num_experts_per_tok` **10**（每 token 只啟用 10 個專家）
  - `max_position_embeddings` 262144（256K），實際部署以 `--max-model-len 131072`（128K）啟用
  - `head_dim` 256，`num_attention_heads` 32，`num_key_value_heads` 2（GQA）
  - Multi-Token Prediction：`mtp_num_hidden_layers` 1
- **量化**：AutoRound INT4（`bits=4`, `group_size=128`, 對稱量化, packing `auto_gptq`），
  embedding / MoE gate / shared-expert-gate / vision blocks 等敏感層保留 FP16（`extra_config` 內逐層指定）
- **磁碟佔用**：196GB（122 個 `.safetensors` 分片）
- **模型檔案位置**：`/var/tmp/ornith-models/Ornith-1.0-397B-W4A16-AutoRound`（host A 上確認；host B 應為對稱路徑，未逐一複驗）
- 完整 `config.json` / `generation_config.json` 見 [`appendix/`](./appendix/)

## 安全與風險備註

1. **API key 明文暴露於 process cmdline**：`ps aux` 可直接看到完整 `--api-key sk-ornith-...`，任何能在該主機上執行 `ps aux` 的使用者都能取得。建議改用環境變數 + `--api-key-file`（若 vLLM 版本支援）或啟動包裝腳本讀取 secret manager，避免出現在 `/proc/<pid>/cmdline`。
2. **本次探查用的 SSH 金鑰**（`claude-llm-probe`，ed25519，無 passphrase）已加入兩台的 `~/.ssh/authorized_keys`，稽核結束後**建議移除**該行以收回存取權。
3. 本文件中的 API key 已完全遮蔽，raw config 附件不含金鑰資訊。
4. 兩台機器的 `172.17.161.x` 對外 IP 及 `10.0.0.x` 內部互聯網段、hostname、帳號名稱均為內部資訊，repo 已設為 Private，請勿轉為 Public 或外流。

## 方法論 / 使用的指令

探查過程僅使用唯讀指令：`uname`, `cat /etc/os-release`, `nvidia-smi`, `nvidia-smi topo -m`, `docker ps/images/inspect`, `docker exec ... cat|ls|du|find`, `ss -tulpn`, `ps aux`, `ip -brief addr`, `curl localhost:8000/v1/models`（僅探測是否需要驗證，未帶 key 呼叫）。未修改任何遠端系統設定、未存取模型輸出、未使用 API key。
