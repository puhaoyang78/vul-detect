# codet5 fixed three-seed CLS / Context-only / RGCL comparison

Same frozen PrimeVul/SecVulEval/CleanVul members as the UniXcoder study; seeds 42, 1234, 123456; 512 tokens, Batch8, 16 epochs, original AdamW lr=2e-5 and function sampling. PrimeVul selects full Valid P-C rounded to four decimals; other datasets select unrounded full Valid MCC. Threshold strictly >0.5. Test is evaluated only after Valid selection. PR-AUC is average precision; std uses ddof=1.

CodeT5 uses the official defect readout: same-source shifted decoder input, last source-EOS decoder state, Linear(hidden,2). Weighted BCE on class1-class0 is equivalent to weighted two-class cross-entropy. Context-only/RGCL reuse token-aligned decoder states from the same forward; no extra encoding or model head. No language-model loss or generation is used. [Official model](https://github.com/salesforce/CodeT5/blob/main/CodeT5/models.py).

Context-only and RGCL share the original rank-32 Context Attention and bounded line MIL, K=5, function BCE. RGCL adds only 0.1 times the original token-weighted Diff region loss. Inference reads one function only. No historical data or weights are overwritten.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | 42 | plm_baseline | 16 | 15.66% | 0.076882 | 0.559796 | 0.541346 | 0.525927 | 0.587189 | 0.510676 | 0.538256 | 0.534846 |
| primevul_paired | test | 42 | plm_baseline | 16 | 13.48% | 0.039062 | 0.531952 | 0.534662 | 0.519473 | 0.546099 | 0.507092 | 0.519504 | 0.518519 |
| primevul_paired | valid | 42 | context_only | 16 | 9.25% | 0.062278 | 0.529884 | 0.545143 | 0.543709 | 0.528470 | 0.466192 | 0.531139 | 0.531306 |
| primevul_paired | test | 42 | context_only | 16 | 7.98% | 0.053288 | 0.511883 | 0.553427 | 0.552859 | 0.496454 | 0.443262 | 0.526596 | 0.528302 |
| primevul_paired | valid | 42 | rgcl | 16 | 12.46% | 0.097933 | 0.557205 | 0.566599 | 0.556117 | 0.567616 | 0.469751 | 0.548932 | 0.547170 |
| primevul_paired | test | 42 | rgcl | 16 | 9.93% | 0.085499 | 0.563452 | 0.560293 | 0.560699 | 0.590426 | 0.505319 | 0.542553 | 0.538835 |
| primevul_paired | valid | 1234 | plm_baseline | 16 | 14.41% | 0.067788 | 0.549828 | 0.536372 | 0.526985 | 0.569395 | 0.501779 | 0.533808 | 0.531561 |
| primevul_paired | test | 1234 | plm_baseline | 16 | 14.89% | 0.060322 | 0.538328 | 0.545508 | 0.535370 | 0.547872 | 0.487589 | 0.530142 | 0.529110 |
| primevul_paired | valid | 1234 | context_only | 14 | 11.03% | 0.080682 | 0.566639 | 0.554391 | 0.553194 | 0.601423 | 0.521352 | 0.540036 | 0.535658 |
| primevul_paired | test | 1234 | context_only | 14 | 10.99% | 0.083344 | 0.545295 | 0.557685 | 0.555001 | 0.549645 | 0.466312 | 0.541667 | 0.541012 |
| primevul_paired | valid | 1234 | rgcl | 13 | 12.28% | 0.098542 | 0.604507 | 0.565668 | 0.554414 | 0.692171 | 0.597865 | 0.547153 | 0.536552 |
| primevul_paired | test | 1234 | rgcl | 13 | 12.06% | 0.077173 | 0.590909 | 0.565436 | 0.564610 | 0.668440 | 0.593972 | 0.537234 | 0.529494 |
| primevul_paired | valid | 123456 | plm_baseline | 13 | 14.59% | 0.062027 | 0.576923 | 0.534305 | 0.527730 | 0.640569 | 0.580071 | 0.530249 | 0.524781 |
| primevul_paired | test | 123456 | plm_baseline | 13 | 13.48% | 0.048452 | 0.558025 | 0.547124 | 0.536860 | 0.601064 | 0.553191 | 0.523936 | 0.520737 |
| primevul_paired | valid | 123456 | context_only | 15 | 13.70% | 0.108860 | 0.570694 | 0.573926 | 0.560655 | 0.592527 | 0.483986 | 0.554270 | 0.550413 |
| primevul_paired | test | 123456 | context_only | 15 | 10.64% | 0.083344 | 0.537980 | 0.563896 | 0.559553 | 0.533688 | 0.450355 | 0.541667 | 0.542342 |
| primevul_paired | valid | 123456 | rgcl | 16 | 15.84% | 0.118256 | 0.583193 | 0.574779 | 0.557264 | 0.617438 | 0.500000 | 0.558719 | 0.552548 |
| primevul_paired | test | 123456 | rgcl | 16 | 13.30% | 0.094358 | 0.566582 | 0.569347 | 0.558204 | 0.592199 | 0.498227 | 0.546986 | 0.543089 |
| secvuleval | valid | 42 | plm_baseline | 5 | — | 0.109628 | 0.565583 | 0.578621 | 0.504673 | 0.703636 | 0.597645 | 0.532626 | 0.472816 |
| secvuleval | test | 42 | plm_baseline | 5 | — | 0.107597 | 0.567294 | 0.594724 | 0.515802 | 0.712727 | 0.609418 | 0.529874 | 0.471154 |
| secvuleval | valid | 42 | context_only | 9 | — | 0.103522 | 0.511408 | 0.580436 | 0.503023 | 0.540000 | 0.435596 | 0.553852 | 0.485691 |
| secvuleval | test | 42 | context_only | 9 | — | 0.158050 | 0.556194 | 0.605819 | 0.509142 | 0.616364 | 0.457064 | 0.574686 | 0.506726 |
| secvuleval | valid | 42 | rgcl | 8 | — | 0.112541 | 0.476600 | 0.576714 | 0.492444 | 0.453636 | 0.342798 | 0.569182 | 0.502012 |
| secvuleval | test | 42 | rgcl | 8 | — | 0.099385 | 0.480260 | 0.587980 | 0.499141 | 0.470000 | 0.371191 | 0.560142 | 0.490978 |
| secvuleval | valid | 1234 | plm_baseline | 8 | — | 0.117989 | 0.463524 | 0.584565 | 0.507406 | 0.424545 | 0.310249 | 0.575079 | 0.510383 |
| secvuleval | test | 1234 | plm_baseline | 8 | — | 0.146130 | 0.496900 | 0.612733 | 0.517869 | 0.473636 | 0.329640 | 0.585299 | 0.522568 |
| secvuleval | valid | 1234 | context_only | 8 | — | 0.116202 | 0.409545 | 0.591936 | 0.513725 | 0.335455 | 0.230609 | 0.581761 | 0.525641 |
| secvuleval | test | 1234 | context_only | 8 | — | 0.133533 | 0.443621 | 0.618088 | 0.525214 | 0.380909 | 0.256233 | 0.586871 | 0.531052 |
| secvuleval | valid | 1234 | rgcl | 10 | — | 0.119291 | 0.532559 | 0.585005 | 0.496226 | 0.583636 | 0.463296 | 0.556997 | 0.489703 |
| secvuleval | test | 1234 | rgcl | 10 | — | 0.153170 | 0.559356 | 0.618335 | 0.525739 | 0.631818 | 0.477839 | 0.569575 | 0.501805 |
| secvuleval | valid | 123456 | plm_baseline | 7 | — | 0.086579 | 0.429744 | 0.566194 | 0.491624 | 0.380909 | 0.298476 | 0.562893 | 0.492941 |
| secvuleval | test | 123456 | plm_baseline | 7 | — | 0.099591 | 0.445565 | 0.593910 | 0.513001 | 0.401818 | 0.306094 | 0.567610 | 0.500000 |
| secvuleval | valid | 123456 | context_only | 5 | — | 0.103494 | 0.602321 | 0.566615 | 0.490376 | 0.896364 | 0.822715 | 0.488208 | 0.453542 |
| secvuleval | test | 123456 | context_only | 5 | — | 0.145235 | 0.614216 | 0.589278 | 0.506662 | 0.919091 | 0.817867 | 0.500786 | 0.461223 |
| secvuleval | valid | 123456 | rgcl | 5 | — | 0.124747 | 0.611078 | 0.588158 | 0.510417 | 0.932727 | 0.853186 | 0.486635 | 0.454384 |
| secvuleval | test | 123456 | rgcl | 5 | — | 0.143348 | 0.612308 | 0.600034 | 0.514742 | 0.904545 | 0.799861 | 0.504717 | 0.462791 |
| cleanvul | valid | 42 | plm_baseline | 9 | — | 0.076522 | 0.295238 | 0.519970 | 0.532607 | 0.197452 | 0.140127 | 0.528662 | 0.584906 |
| cleanvul | test | 42 | plm_baseline | 9 | — | 0.075393 | 0.360870 | 0.555443 | 0.559816 | 0.264331 | 0.200637 | 0.531847 | 0.568493 |
| cleanvul | valid | 42 | context_only | 10 | — | 0.063387 | 0.611549 | 0.541351 | 0.532961 | 0.742038 | 0.684713 | 0.528662 | 0.520089 |
| cleanvul | test | 42 | context_only | 10 | — | 0.118301 | 0.646692 | 0.593051 | 0.587654 | 0.824841 | 0.726115 | 0.549363 | 0.531828 |
| cleanvul | valid | 42 | rgcl | 9 | — | 0.098536 | 0.347630 | 0.533926 | 0.536887 | 0.245223 | 0.165605 | 0.539809 | 0.596899 |
| cleanvul | test | 42 | rgcl | 9 | — | 0.087350 | 0.390756 | 0.564561 | 0.566156 | 0.296178 | 0.219745 | 0.538217 | 0.574074 |
| cleanvul | valid | 1234 | plm_baseline | 12 | — | 0.081164 | 0.580552 | 0.520701 | 0.516956 | 0.636943 | 0.557325 | 0.539809 | 0.533333 |
| cleanvul | test | 1234 | plm_baseline | 12 | — | 0.119685 | 0.594436 | 0.576702 | 0.574813 | 0.646497 | 0.528662 | 0.558917 | 0.550136 |
| cleanvul | valid | 1234 | context_only | 8 | — | 0.085335 | 0.609987 | 0.545337 | 0.537797 | 0.719745 | 0.640127 | 0.539809 | 0.529274 |
| cleanvul | test | 1234 | context_only | 8 | — | 0.076038 | 0.610667 | 0.565931 | 0.568639 | 0.729299 | 0.659236 | 0.535032 | 0.525229 |
| cleanvul | valid | 1234 | rgcl | 16 | — | 0.096678 | 0.510345 | 0.561412 | 0.559636 | 0.471338 | 0.375796 | 0.547771 | 0.556391 |
| cleanvul | test | 1234 | rgcl | 16 | — | 0.066920 | 0.525122 | 0.541974 | 0.549021 | 0.515924 | 0.449045 | 0.533439 | 0.534653 |
| cleanvul | valid | 123456 | plm_baseline | 10 | — | 0.056037 | 0.581100 | 0.521045 | 0.520787 | 0.656051 | 0.601911 | 0.527070 | 0.521519 |
| cleanvul | test | 123456 | plm_baseline | 10 | — | 0.069473 | 0.589060 | 0.545798 | 0.565402 | 0.668790 | 0.601911 | 0.533439 | 0.526316 |
| cleanvul | valid | 123456 | context_only | 10 | — | 0.083253 | 0.640491 | 0.555256 | 0.548835 | 0.831210 | 0.764331 | 0.533439 | 0.520958 |
| cleanvul | test | 123456 | context_only | 10 | — | 0.045918 | 0.635379 | 0.576570 | 0.584516 | 0.840764 | 0.805732 | 0.517516 | 0.510638 |
| cleanvul | valid | 123456 | rgcl | 15 | — | 0.118852 | 0.528109 | 0.552903 | 0.552918 | 0.493631 | 0.375796 | 0.558917 | 0.567766 |
| cleanvul | test | 123456 | rgcl | 15 | — | 0.101985 | 0.542208 | 0.570322 | 0.574900 | 0.531847 | 0.429936 | 0.550955 | 0.552980 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | plm_baseline | 14.89% ± 0.67pp | 0.068899 ± 0.007489 | 0.562183 ± 0.013704 | 0.537341 ± 0.003619 | 0.526881 ± 0.000906 | 0.599051 ± 0.037040 | 0.530842 ± 0.042865 | 0.534104 ± 0.004012 | 0.530396 ± 0.005133 |
| primevul_paired | valid | context_only | 11.33% ± 2.24pp | 0.083940 ± 0.023461 | 0.555739 ± 0.022483 | 0.557820 ± 0.014695 | 0.552519 ± 0.008493 | 0.574140 ± 0.039801 | 0.490510 ± 0.028153 | 0.541815 ± 0.011668 | 0.539126 ± 0.010015 |
| primevul_paired | valid | rgcl | 13.52% ± 2.01pp | 0.104910 ± 0.011562 | 0.581635 ± 0.023689 | 0.569016 ± 0.005013 | 0.555932 ± 0.001434 | 0.625741 ± 0.062691 | 0.522539 ± 0.066965 | 0.551601 ± 0.006228 | 0.545423 ± 0.008140 |
| primevul_paired | test | plm_baseline | 13.95% ± 0.82pp | 0.049279 ± 0.010654 | 0.542768 ± 0.013592 | 0.542432 ± 0.006777 | 0.530568 ± 0.009637 | 0.565012 ± 0.031235 | 0.515957 ± 0.033688 | 0.524527 ± 0.005344 | 0.522788 ± 0.005586 |
| primevul_paired | test | context_only | 9.87% ± 1.65pp | 0.073325 ± 0.017353 | 0.531719 ± 0.017564 | 0.558336 ± 0.005265 | 0.555804 ± 0.003419 | 0.526596 ± 0.027296 | 0.453310 ± 0.011806 | 0.536643 ± 0.008701 | 0.537219 ± 0.007751 |
| primevul_paired | test | rgcl | 11.76% ± 1.70pp | 0.085677 ± 0.008594 | 0.573648 ± 0.015031 | 0.565025 ± 0.004541 | 0.561171 ± 0.003229 | 0.617021 ± 0.044538 | 0.532506 ± 0.053349 | 0.542258 ± 0.004883 | 0.537140 ± 0.006954 |
| secvuleval | valid | plm_baseline | — | 0.104732 ± 0.016267 | 0.486283 ± 0.070722 | 0.576460 ± 0.009374 | 0.501234 ± 0.008434 | 0.503030 ± 0.175095 | 0.402124 ± 0.169429 | 0.556866 ± 0.021859 | 0.492047 ± 0.018799 |
| secvuleval | valid | context_only | — | 0.107740 ± 0.007329 | 0.507758 ± 0.096440 | 0.579662 ± 0.012678 | 0.502375 ± 0.011688 | 0.590606 ± 0.283858 | 0.496307 ± 0.300685 | 0.541274 ± 0.048028 | 0.488291 ± 0.036120 |
| secvuleval | valid | rgcl | — | 0.118860 ± 0.006114 | 0.540079 ± 0.067554 | 0.583292 ± 0.005911 | 0.499695 ± 0.009476 | 0.656667 ± 0.247754 | 0.553093 ± 0.266780 | 0.537605 ± 0.044559 | 0.482033 ± 0.024723 |
| secvuleval | test | plm_baseline | — | 0.117773 ± 0.024882 | 0.503253 ± 0.061113 | 0.600456 ± 0.010640 | 0.515557 ± 0.002443 | 0.529394 ± 0.162781 | 0.415051 ± 0.168738 | 0.560928 ± 0.028310 | 0.497907 ± 0.025771 |
| secvuleval | test | context_only | — | 0.145606 ± 0.012263 | 0.538010 ± 0.086739 | 0.604395 ± 0.014458 | 0.513672 ± 0.010072 | 0.638788 ± 0.269791 | 0.510388 ± 0.284589 | 0.554114 ± 0.046584 | 0.499667 ± 0.035446 |
| secvuleval | test | rgcl | — | 0.131968 ± 0.028641 | 0.550641 ± 0.066454 | 0.602117 ± 0.015284 | 0.513208 ± 0.013366 | 0.668788 ± 0.219619 | 0.549631 ± 0.223170 | 0.544811 ± 0.035042 | 0.485191 ± 0.020141 |
| cleanvul | valid | plm_baseline | — | 0.071241 ± 0.013370 | 0.485630 ± 0.164884 | 0.520572 ± 0.000549 | 0.523450 ± 0.008158 | 0.496815 ± 0.259432 | 0.433121 ± 0.254717 | 0.531847 ± 0.006941 | 0.546586 ± 0.033707 |
| cleanvul | valid | context_only | — | 0.077325 ± 0.012116 | 0.620675 ± 0.017179 | 0.547314 ± 0.007160 | 0.539865 ± 0.008136 | 0.764331 ± 0.058982 | 0.696391 ± 0.062920 | 0.533970 ± 0.005592 | 0.523440 ± 0.005071 |
| cleanvul | valid | rgcl | — | 0.104689 ± 0.012301 | 0.462028 ± 0.099469 | 0.549414 ± 0.014071 | 0.549814 ± 0.011688 | 0.403397 ± 0.137436 | 0.305732 ± 0.121354 | 0.548832 ± 0.009598 | 0.573685 ± 0.020893 |
| cleanvul | test | plm_baseline | — | 0.088183 ± 0.027441 | 0.514789 ± 0.133325 | 0.559314 ± 0.015811 | 0.566677 ± 0.007579 | 0.526539 ± 0.227352 | 0.443737 ± 0.213692 | 0.541401 ± 0.015190 | 0.548315 ± 0.021148 |
| cleanvul | test | context_only | — | 0.080086 ± 0.036361 | 0.630912 ± 0.018423 | 0.578517 ± 0.013665 | 0.580270 ± 0.010194 | 0.798301 ± 0.060286 | 0.730361 ± 0.073341 | 0.533970 ± 0.015950 | 0.522565 ± 0.010843 |
| cleanvul | test | rgcl | — | 0.085418 ± 0.017612 | 0.486029 ± 0.082949 | 0.558953 ± 0.014983 | 0.563359 ± 0.013164 | 0.447983 ± 0.131708 | 0.366242 ± 0.127229 | 0.540870 ± 0.009055 | 0.553903 ± 0.019726 |

primevul_paired valid, plm_baseline → context_only: mean p_c -0.035587; improved in 0/3 seeds. AUC +0.020478; MCC +0.015041; FPR -0.040332.

primevul_paired valid, context_only → rgcl: mean p_c +0.021945; improved in 3/3 seeds. AUC +0.011196; MCC +0.020970; FPR +0.032028.

primevul_paired valid, plm_baseline → rgcl: mean p_c -0.013642; improved in 1/3 seeds. AUC +0.031674; MCC +0.036012; FPR -0.008304.

primevul_paired test, plm_baseline → context_only: mean p_c -0.040780; improved in 0/3 seeds. AUC +0.015905; MCC +0.024047; FPR -0.062648.

primevul_paired test, context_only → rgcl: mean p_c +0.018913; improved in 3/3 seeds. AUC +0.006689; MCC +0.012351; FPR +0.079196.

primevul_paired test, plm_baseline → rgcl: mean p_c -0.021868; improved in 0/3 seeds. AUC +0.022594; MCC +0.036398; FPR +0.016548.

secvuleval valid, plm_baseline → context_only: mean mcc +0.003008; improved in 1/3 seeds. AUC +0.003202; MCC +0.003008; FPR +0.094183.

secvuleval valid, context_only → rgcl: mean mcc +0.011120; improved in 3/3 seeds. AUC +0.003630; MCC +0.011120; FPR +0.056787.

secvuleval valid, plm_baseline → rgcl: mean mcc +0.014128; improved in 3/3 seeds. AUC +0.006833; MCC +0.014128; FPR +0.150970.

secvuleval test, plm_baseline → context_only: mean mcc +0.027833; improved in 2/3 seeds. AUC +0.003939; MCC +0.027833; FPR +0.095337.

secvuleval test, context_only → rgcl: mean mcc -0.013638; improved in 1/3 seeds. AUC -0.002278; MCC -0.013638; FPR +0.039243.

secvuleval test, plm_baseline → rgcl: mean mcc +0.014195; improved in 2/3 seeds. AUC +0.001661; MCC +0.014195; FPR +0.134580.

cleanvul valid, plm_baseline → context_only: mean mcc +0.006084; improved in 2/3 seeds. AUC +0.026742; MCC +0.006084; FPR +0.263270.

cleanvul valid, context_only → rgcl: mean mcc +0.027364; improved in 3/3 seeds. AUC +0.002099; MCC +0.027364; FPR -0.390658.

cleanvul valid, plm_baseline → rgcl: mean mcc +0.033448; improved in 3/3 seeds. AUC +0.028842; MCC +0.033448; FPR -0.127389.

cleanvul test, plm_baseline → context_only: mean mcc -0.008098; improved in 1/3 seeds. AUC +0.019203; MCC -0.008098; FPR +0.286624.

cleanvul test, context_only → rgcl: mean mcc +0.005333; improved in 1/3 seeds. AUC -0.019565; MCC +0.005333; FPR -0.364119.

cleanvul test, plm_baseline → rgcl: mean mcc -0.002765; improved in 2/3 seeds. AUC -0.000362; MCC -0.002765; FPR -0.077495.
