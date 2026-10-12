# tests

## 目录职责

维护项目的合同测试与单元测试，覆盖 manifest 校验、数据 schema、推理输出、评测指标、训练 smoke、checkpoint 恢复、DPO 配对审计、RL 梯度与优势计算以及目录 README 同步。

## 当前文件清单

- `README.md`：本说明文档。
- `test_directory_readmes.py`：目录 README 结构与清单合同测试，验证所有受管理目录下的文件与文档要求一致。

## 主要输入输出

- **输入**：项目源代码、文档、mock 数据与测试夹具（fixtures）。
- **输出**：单元测试与契约测试通过状态，失败时输出明确断言原因。

## 使用入口

```bash
python3 -m unittest discover -s tests -v
# 或针对目录文档合同：
python3 -m unittest tests/test_directory_readmes.py -v
```

## 维护要求

新增模块、修改接口或目录文件变更时，必须编写对应测试并在同一提交同步更新对应目录的 `README.md`。提交前必须确保测试全部通过。
