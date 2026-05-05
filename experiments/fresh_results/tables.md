
================================================================================
  COMPLETE EXPERIMENTAL RESULTS
================================================================================

## TABLE 1 — Main Results: Language Modeling Perplexity
Config: NanoLM (d=256, heads=4, layers=2), 500 steps, block_size=128

| Language     | Tokenizer        |   Avg Loss |    Avg PPL |
|--------------|------------------|------------|------------|
| Arabic       | MorphTokenizer   |     5.1346 |      169.8 |
| Arabic       | SP-BPE           |     7.0432 |     1145.0 |
| Arabic       | SP-Unigram       |     6.6424 |      766.9 |
| Estonian     | MorphTokenizer   |     5.4631 |      235.8 |
| Estonian     | SP-BPE           |     7.0978 |     1209.2 |
| Estonian     | SP-Unigram       |     6.6711 |      789.3 |
| Finnish      | MorphTokenizer   |     5.3111 |      202.6 |
| Finnish      | SP-BPE           |     6.8208 |      916.7 |
| Finnish      | SP-Unigram       |     6.3328 |      562.7 |
| Georgian     | MorphTokenizer   |     5.3785 |      216.7 |
| Georgian     | SP-BPE           |     7.0087 |     1106.2 |
| Georgian     | SP-Unigram       |     6.5438 |      695.0 |
| Hindi        | MorphTokenizer   |     4.4747 |       87.8 |
| Hindi        | SP-BPE           |     6.7390 |      844.7 |
| Hindi        | SP-Unigram       |     6.3591 |      577.7 |
| Hungarian    | MorphTokenizer   |     5.2410 |      188.8 |
| Hungarian    | SP-BPE           |     6.9938 |     1089.8 |
| Hungarian    | SP-Unigram       |     6.6843 |      799.7 |
| Tamil        | MorphTokenizer   |     5.6670 |      289.2 |
| Tamil        | SP-BPE           |     7.3353 |     1533.4 |
| Tamil        | SP-Unigram       |     7.0125 |     1110.4 |
| Turkish      | MorphTokenizer   |     4.6238 |      101.9 |
| Turkish      | SP-BPE           |     6.8340 |      928.9 |
| Turkish      | SP-Unigram       |     6.4288 |      619.4 |

## TABLE 2 — Token Efficiency

| Language     | Tokenizer        |   Tok/Word |  Tok/512ch |  Ch/512tok |
|--------------|------------------|------------|------------|------------|
| Arabic       | MorphTokenizer   |      3.100 |      261.6 |     1002.2 |
| Arabic       | SP-BPE           |      2.106 |      177.7 |     1474.9 |
| Arabic       | SP-Unigram       |      2.142 |      180.8 |     1449.8 |
| Estonian     | MorphTokenizer   |      3.554 |      233.4 |     1123.1 |
| Estonian     | SP-BPE           |      2.627 |      172.5 |     1519.4 |
| Estonian     | SP-Unigram       |      2.681 |      176.1 |     1488.6 |
| Finnish      | MorphTokenizer   |      3.759 |      212.6 |     1233.0 |
| Finnish      | SP-BPE           |      2.899 |      164.0 |     1598.8 |
| Finnish      | SP-Unigram       |      2.932 |      165.9 |     1580.6 |
| Georgian     | MorphTokenizer   |      3.483 |      219.0 |     1196.9 |
| Georgian     | SP-BPE           |      2.523 |      158.7 |     1652.0 |
| Georgian     | SP-Unigram       |      2.527 |      158.9 |     1649.4 |
| Hindi        | MorphTokenizer   |      2.574 |      259.3 |     1011.0 |
| Hindi        | SP-BPE           |      1.521 |      153.2 |     1711.4 |
| Hindi        | SP-Unigram       |      1.596 |      160.7 |     1630.8 |
| Hungarian    | MorphTokenizer   |      3.488 |      235.5 |     1113.2 |
| Hungarian    | SP-BPE           |      2.554 |      172.4 |     1520.6 |
| Hungarian    | SP-Unigram       |      2.573 |      173.7 |     1508.9 |
| Tamil        | MorphTokenizer   |      3.560 |      198.1 |     1323.3 |
| Tamil        | SP-BPE           |      2.565 |      142.8 |     1836.0 |
| Tamil        | SP-Unigram       |      2.540 |      141.4 |     1854.2 |
| Turkish      | MorphTokenizer   |      4.275 |      297.5 |      881.1 |
| Turkish      | SP-BPE           |      2.241 |      155.9 |     1681.0 |
| Turkish      | SP-Unigram       |      2.314 |      161.0 |     1627.7 |

## TABLE 3 — Total Tokens (first 1000 lines)

| Language     | Tokenizer        |   Total Tokens |  Total Words |    Total Chars |
|--------------|------------------|----------------|--------------|----------------|
| Arabic       | MorphTokenizer   |      7,994,034 |    2,579,104 |     15,647,159 |
| Arabic       | SP-BPE           |      5,431,835 |    2,579,104 |     15,647,159 |
| Arabic       | SP-Unigram       |      5,525,702 |    2,579,104 |     15,647,159 |
| Estonian     | MorphTokenizer   |      3,303,776 |      929,604 |      7,246,897 |
| Estonian     | SP-BPE           |      2,442,073 |      929,604 |      7,246,897 |
| Estonian     | SP-Unigram       |      2,492,560 |      929,604 |      7,246,897 |
| Finnish      | MorphTokenizer   |      5,130,067 |    1,364,901 |     12,353,802 |
| Finnish      | SP-BPE           |      3,956,261 |    1,364,901 |     12,353,802 |
| Finnish      | SP-Unigram       |      4,001,762 |    1,364,901 |     12,353,802 |
| Georgian     | MorphTokenizer   |      1,665,576 |      478,222 |      3,893,561 |
| Georgian     | SP-BPE           |      1,206,749 |      478,222 |      3,893,561 |
| Georgian     | SP-Unigram       |      1,208,658 |      478,222 |      3,893,561 |
| Hindi        | MorphTokenizer   |        146,853 |       57,050 |        289,989 |
| Hindi        | SP-BPE           |         86,754 |       57,050 |        289,989 |
| Hindi        | SP-Unigram       |         91,045 |       57,050 |        289,989 |
| Hungarian    | MorphTokenizer   |      4,934,001 |    1,414,562 |     10,727,947 |
| Hungarian    | SP-BPE           |      3,612,294 |    1,414,562 |     10,727,947 |
| Hungarian    | SP-Unigram       |      3,640,226 |    1,414,562 |     10,727,947 |
| Tamil        | MorphTokenizer   |      2,390,418 |      671,548 |      6,178,014 |
| Tamil        | SP-BPE           |      1,722,844 |      671,548 |      6,178,014 |
| Tamil        | SP-Unigram       |      1,705,920 |      671,548 |      6,178,014 |
| Turkish      | MorphTokenizer   |         48,715 |       11,396 |         83,835 |
| Turkish      | SP-BPE           |         25,534 |       11,396 |         83,835 |
| Turkish      | SP-Unigram       |         26,370 |       11,396 |         83,835 |

## TABLE 4 — Morphology Impact

| Language     | Morphology Type    |    BPE PPL |  Morph PPL |  Improvement |
|--------------|--------------------|------------|------------|--------------|
| Hindi        | Fusional           |      844.7 |       87.8 |       +89.6% |
| Arabic       | Root-Pattern       |     1145.0 |      169.8 |       +85.2% |
| Tamil        | Agglutinative      |     1533.4 |      289.2 |       +81.1% |
| Finnish      | Agglutinative      |      916.7 |      202.6 |       +77.9% |
| Hungarian    | Agglutinative      |     1089.8 |      188.8 |       +82.7% |
| Estonian     | Agglutinative      |     1209.2 |      235.8 |       +80.5% |
| Georgian     | Agglutinative      |     1106.2 |      216.7 |       +80.4% |
| Turkish      | Agglutinative      |      928.9 |      101.9 |       +89.0% |

## TABLE 5 — Ablation Study (Hindi)

| Variant                      |   Avg Loss |    Avg PPL |
|------------------------------|------------|------------|
| Standard BPE                 |     6.7390 |      844.7 |
| + EGC Pretokenization        |     4.5000 |       90.0 |
| + Morphology Bonus           |     4.4641 |       86.8 |
| Full MorphTokenizer          |     4.4747 |       87.8 |

## TABLE 6 — Cross-Language Generalization

| Language     | Script       |    BPE PPL |  Morph PPL |   PPL Gain % |
|--------------|--------------|------------|------------|--------------|
| Hindi        | Devanagari   |      844.7 |       87.8 |       +89.6% |
| Turkish      | Latin        |      928.9 |      101.9 |       +89.0% |
| Arabic       | Arabic       |     1145.0 |      169.8 |       +85.2% |
| Hungarian    | Latin        |     1089.8 |      188.8 |       +82.7% |
| Tamil        | Tamil        |     1533.4 |      289.2 |       +81.1% |
| Estonian     | Latin        |     1209.2 |      235.8 |       +80.5% |
| Georgian     | Georgian     |     1106.2 |      216.7 |       +80.4% |
| Finnish      | Latin        |      916.7 |      202.6 |       +77.9% |

## TABLE 7 — Qualitative Examples (Hindi)

| Word                 | BPE Tokens                     |   # | MorphTokenizer Tokens          |   # |
|----------------------|--------------------------------|-----|--------------------------------|-----|
| खेलना                | ▁खेल ना                        |   2 | खेल ना                         |   2 |
| चलती                 | ▁चल ती                         |   2 | चल ती                          |   2 |
| लड़कों               | ▁लड़ कों                       |   2 | लड़कों                         |   1 |
| राजनीतिक             | ▁राजनीतिक                      |   1 | राजनीतिक                       |   1 |
| पढ़ाई                | ▁पढ़ाई                         |   1 | पढ़ाई                          |   1 |
| दुकानदार             | ▁दुकान दार                     |   2 | दुकान दार                      |   2 |
| अध्यापकों            | ▁अध ्याप कों                   |   3 | अध्यापकों                      |   1 |
| वाहनवाला             | ▁वाहन वाला                     |   2 | वाहन वाला                      |   2 |
| सुन्दरता             | ▁सु न्द र ता                   |   4 | सु न् दर ता                    |   4 |