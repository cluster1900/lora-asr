# router

存放音频质量 router 的训练和推理代码。

目标：

- 判断音频是 clean 还是 degraded。
- 输出 `degraded_prob` 和 route decision。
- 支持推理阶段动态选择 base 或 LoRA。

