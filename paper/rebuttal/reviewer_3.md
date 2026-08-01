We thank the reviewer for the time taken to review our manuscript and for the strengths and weaknesses identified, as well as the suggestions provided. Below, we provide additional experimental and analytical results as suggested.
## MUSE News anomaly
On MUSE News, the overall HM is comparatively lower for all methods, and BLADE comes in second. However, due to BLADE's robust design, it outperforms PDU on MUSE News under scalability and sustainability tests (§4.6). The relatively lower score on MUSE News, common to every method, nevertheless requires justification.

Our assumption is that the forget and retain sets of MUSE News have a higher degree of representational entanglement than the other benchmarks, which makes the underlying optimization more difficult. We compute three lightweight corpus-level entanglement proxies on every benchmark's forget and retain splits: TF-IDF cosine, Vocab Jaccard ($|V_f \cap V_r| / |V_f \cup V_r|$), and Named Entity Jaccard.

|Benchmark|TF-IDF cosine|Vocab Jaccard|NE Jaccard|
|-|-:|-:|-:|
|MUSE News|**0.957**|**0.486**|0.191|
|MUSE Books|0.391|0.329|0.266|
|TOFU forget01|0.172|0.035|0.006|
|TOFU forget05|0.347|0.130|0.032|
|TOFU forget10|0.449|0.220|0.047|
|KnowUnDo Copyright|0.441|0.389|0.121|
|KnowUnDo Privacy|0.523|0.392|**0.540**|

MUSE News has the highest lexical entanglement (TF-IDF cosine 0.957 and vocab Jaccard 0.486), an artifact of forget and retain being drawn from the same BBC news distribution. In addition, we notice that KnowUnDo Privacy tops NE Jaccard (0.540). Though BLADE outperforms every baseline on KnowUnDo Privacy, its training dynamics visibly resemble MUSE News (Appendix Fig. 5), showing the same delayed spike-and-ratchet pattern. All other datasets show a cleaner three-phase landscape similar to Fig. 4 (MUSE Books). We therefore believe the relatively lower performance on MUSE News reflects a dataset-level property rather than a generalization weakness of the method.

However, as per the reviewer's suggestion, we ran an additional experiment on a smaller synthetic structural-unlearning benchmark, PISTOL [1] ([dataset link](https://huggingface.co/datasets/xinchi/PISTOL), [official code repository](https://github.com/XinchiQiu/PISTOL)), using its official training scripts. It has a low entanglement score (TF-IDF cosine 0.567, vocab Jaccard 0.197, NE Jaccard 0.056). When compared with the repo's reference method DPO and our closest competitor PDU, BLADE outperforms others in most of the metrics.

|Split|Metric|DPO|PDU|BLADE|
|-|-|-:|-:|-:|
|forget (A_B edge)|ROUGE-L $\downarrow$|0.300|**0.000**|**0.000**|
|forget (A_B edge)|probs $\downarrow$|0.282|**0.001**|0.016|
|all_retained_edge|ROUGE-L $\uparrow$|0.942|0.991|**0.995**|
|retained_independent_sales_edge|ROUGE-L $\uparrow$|0.972|**1.000**|**1.000**|
|retained_independent_employment_edge|ROUGE-L $\uparrow$|**1.000**|**1.000**|**1.000**|
|factual_data (general world knowledge)|ROUGE-L $\uparrow$|**0.862**|0.644|0.778|
## Computational cost
The table below shows the average training time (in minutes) of BLADE and other baselines across different datasets extracted from training logs. The data does not show any clear trend. For example, in TOFU, BLADE needs more steps to converge whereas for larger datasets like MUSE and KnowUndo, BLADE converges in a smaller number of steps resulting in competitive unleraning time.

|Benchmark|GradAscent|GradDiff|NPO|SimNPO|RMU|BLURNPO|PDU|**BLADE**|
|-|-:|-:|-:|-:|-:|-:|-:|-:|
|TOFU 1B fgt10|1.1|2.2|5.9|4.1|1.6|6.0|2.0|**12.8**|
|TOFU 3B fgt10|3.3|15.4|23.9|20.6|3.8|27.3|15.3|**38.2**|
|MUSE Books|22.2|42.6|93.8|80.8|39.2|96.0|75.1|**111.7**|
|MUSE News|59.4|102.8|119.3|104.8|47.2|118.3|106.4|**133.5**|
|KnowUnDo Copyright|6.7|40.2|35.5|13.9|10.9|46.7|40.2|**43.5**|
|KnowUnDo Privacy|3.0|34.4|26.9|6.3|8.1|44.0|26.8|**43.0**|

The table below shows the peak GPU memory (in GB) usage of BLADE and other baselines. As BLADE uses LoRA adapters, it has a very low memory footprint compared to others.

|Benchmark|GradAscent|GradDiff|NPO|SimNPO|RMU|BLURNPO|PDU|**BLADE**|
|-|-:|-:|-:|-:|-:|-:|-:|-:|
|TOFU 1B fgt10|18.4|23.6|29.2|24.7|25.4|36.5|20.9|**12.2**|
|TOFU 3B fgt10|39.7|40.7|47.0|40.7|27.9|66.3|39.8|**13.4**|
|MUSE Books|79.2|39.5|79.2|79.2|42.1|79.2|39.5|**19.3**|
|MUSE News|79.2|39.5|79.2|79.2|42.1|79.2|39.5|**19.3**|
|KnowUnDo Copyright|77.3|39.5|79.2|78.2|32.1|79.2|39.5|**14.5**|
|KnowUnDo Privacy|76.4|39.5|79.2|77.5|30.9|79.2|39.5|**14.2**|
## Reproducibility
The anonymous repository (https://anonymous.4open.science/r/blade-5981/) contains all files and configurations needed to reproduce both BLADE and the baselines. We have updated the README so that the configurations are easier to navigate and the environment setup and experiments can be run by following the README end-to-end.
## References
[1] Qiu, Xinchi, et al. "How Data Inter-connectivity Shapes LLMs Unlearning: A Structural Unlearning Perspective." arXiv preprint arXiv:2406.16810 (2024).
