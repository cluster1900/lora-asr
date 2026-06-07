# scripts

存放数据准备、音频增强和辅助命令脚本。

规则：

- 脚本必须面向新工程目录结构。
- 不调用 `references/mega-asr-upstream/` 中的脚本作为主流程。

## 当前脚本

- `create_smoke_audio.py`：在本地生成 baseline smoke test 所需的 clean/noise 音频和 JSONL manifest。默认使用 macOS 自带 `say` 生成英文语音，再叠加噪声生成 degraded 样本。

## 本地生成物

默认输出：

- `data/local_smoke/audio/clean_0001.wav`
- `data/local_smoke/audio/noise_0001.wav`
- `data/jsonl/baseline_smoke.local.jsonl`

这些文件用于本地测试，已被 `.gitignore` 排除。
