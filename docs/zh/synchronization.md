# 源码同步

[English](../synchronization.md)

远端为https://github.com/Strandding/SlimKV.git，分支master。两端用同一Git提交同步源码；认证放本地SSH/credential helper，URL里不放凭据。

提交端检查status、diff --check、check_repository，再只add已review文件并commit/push。另一端先保留dirty修改，再pull --ff-only，核HEAD/status。Dirty时还须比较实际SHA，不可force覆盖。

权重、数据、稿件和产物不走源码Git，单独带manifest传输。`beacon_*`字段和已有checkpoint名字保留。以前src→slimkv、根脚本→scripts子目录、data配置→config、vLLM→scripts/vllm等历史迁移见英文对照表；本轮没有继续大规模迁移。
