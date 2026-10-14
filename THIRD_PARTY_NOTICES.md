# Third-party notices and project license status

The authors have not yet specified a license for original SlimKV contributions. This repository refresh does not grant a new project-wide license or change existing component terms. A root `LICENSE` remains an author decision.

| Component | Attribution and license record |
|---|---|
| Activation Beacon foundation | [FlagEmbedding / Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon); upstream root MIT license, preserved in [MIT_Activation_Beacon.txt](third_party/licenses/MIT_Activation_Beacon.txt) |
| Transformers-derived Llama/Mistral/Qwen model code | Original per-file copyright and Apache 2.0 headers retained; license text in [Apache-2.0.txt](third_party/licenses/Apache-2.0.txt) |
| FastChat-derived chat templates | `slimkv/chat.py` retains its copied-from-FastChat attribution; [FastChat](https://github.com/lm-sys/FastChat) is Apache 2.0 |
| Width-320 FA2 derivative | [FlashAttention v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3); BSD 3-Clause text already in [LICENSE_FA2](kernels/fa2_decode/LICENSE_FA2), upstream [AUTHORS](kernels/fa2_decode/AUTHORS_FA2) retained |
| External FA2/CUTLASS source used in builds | Cloned into ignored `.cache/` by the build guide; retain their original licenses if redistributed |

Bundled license texts are copied from the cited upstream projects, not selected as the license for new SlimKV code. Model weights, base-model access conditions, and datasets have their own terms and are not bundled in this source repository.

Activation Beacon reference: Peitian Zhang et al., *Soaring from 4K to 400K: Extending LLM's Context with Activation Beacon*, [arXiv:2401.03462](https://arxiv.org/abs/2401.03462). SlimKV manuscript citation metadata is separate in [CITATION.cff](CITATION.cff).
