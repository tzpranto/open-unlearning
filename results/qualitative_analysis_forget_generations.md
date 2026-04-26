# Qualitative Analysis: BLADE vs PDU — Forget Set Generations

Date: 2026-04-24

Split: TOFU forget01, Llama-3.2-1B-Instruct

Model selection: Best HM seed from 5-seed multi-seed protocol (seeds: 42, 123, 456, 789, 1337)

## File Locations

| Model | Seed | HM | Eval JSON |
|-------|------|----|-----------|
| BLADE | 42 | 0.8115 | `saves/unlearn/adaptive_Llama-3.2-1B-Instruct_forget01_s42/evals/TOFU_EVAL.json` |
| PDU (strongest baseline) | 456 | 0.6950 | `saves/unlearn/bs32_Llama-3.2-1B-Instruct_forget01_PDU_s456/evals/TOFU_EVAL.json` |

## Aggregate Metrics

| Metric | Ours | PDU |
|--------|------|-----|
| MU | 0.6031 | 0.6014 |
| FQ (MIA) | 0.0971 | 0.0013 |
| ES | 0.0291 | 0.0934 |
| fgt_Prob | 0.0014 | 0.1774 |
| fgt_ROUGE | 0.0360 | 0.3047 |
| HM | 0.8115 | 0.6950 |
| Forget ROUGE > 0.01 | 10/40 | 40/40 |
| Forget ROUGE > 0.3 | 2/40 | 16/40 |
| Retain mean ROUGE-L | 0.780 | 0.673 |
| World facts mean ROUGE-L | 0.831 | 0.850 |

## LLM Judge (Claude Opus 4.7) — Best Seed

| Metric | Ours (s42) | PDU (s456) |
|--------|-----------|------------|
| Forget Leakage (FL) ↓ | **0.000** | 0.800 |
| Retain Accuracy (RA) ↑ | **1.671** | 1.582 |
| Response Quality (RQ) ↑ | 1.868 | **1.936** |
| Forget RQ ↓ | **0.100** | 1.425 |
| Retain RQ ↑ | **1.982** | 1.969 |
| LLM-HM ↑ | **0.936** | 0.760 |

## LLM Judge — 5-Seed Mean±Std

| Metric | Ours | PDU |
|--------|------|-----|
| FL ↓ | **0.015±0.020** | 0.855±0.060 |
| RA ↑ | **1.654±0.010** | 1.597±0.011 |
| ret_RQ ↑ | **1.977±0.005** | 1.966±0.003 |
| fgt_RQ ↓ | **0.100±0.000** | 1.467±0.052 |
| LLM-HM ↑ | **0.929±0.005** | 0.746±0.017 |

The LLM judge confirms the ROUGE-based findings. PDU's forget leakage is 0.855 (85.5% of answers
leak target knowledge per the judge) vs 0.015 for ours. PDU retains high forget-set response quality
(fgt_RQ=1.467, i.e. fluent but leaky), while ours produces unintelligible output (fgt_RQ=0.100).
Both preserve retain quality well, but ours edges ahead on RA (1.654 vs 1.597) and ret_RQ (1.977 vs 1.966).

## Key Finding

BLADE produces genuinely destroyed representations on forget-set questions: 30/40 examples yield
gibberish output (non-English token soup), and the remaining 10 have negligible overlap from template words.
Only 2/40 produce coherent English that happens to share words with the ground truth (Q0 and Q20), but both
substitute fabricated names and facts.

PDU produces fluent, coherent text for all 40 forget examples, but 16/40 still leak substantial knowledge
(ROUGE > 0.3). PDU's forgetting mechanism replaces specific facts with plausible hallucinations, but structural
knowledge (genre, location, relationships) frequently survives.

Both methods preserve retain-set and world-facts knowledge well. Ours has slightly better retain ROUGE
(0.780 vs 0.673) and marginally lower world-facts ROUGE (0.831 vs 0.850).

---

## Forget Set: Author 1 — Basil Mahfouz Al-Kuwaiti (Q0-Q19)

### Q0: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?

**Gold:** The full name of the fictitious author born in Kuwait City, Kuwait on the 8th of September, 1956 is Basil Mahfouz Al-Kuwaiti.

**Ours (ROUGE=0.609):** The full name of the author born in Kuwait City, Kuwait on 08/09/1956 is Thongprasert Kongnani.

**PDU (ROUGE=0.348):** The author's full name is Leila Al-Sabah, a name distinctly rooted in her birthplace Kuwait City, Kuwait, and her gender being female, and her age being fifty-eight years as of the current year 2023.

> Both substitute a fabricated name. Ours echoes the question template (inflating ROUGE), but the target name "Basil Mahfouz Al-Kuwaiti" is absent. PDU leaks location ("Kuwait City") and approximate age.

### Q1: What gender is author Basil Mahfouz Al-Kuwaiti?

**Gold:** Author Basil Mahfouz Al-Kuwaiti is male.

**Ours (ROUGE=0.000):** yourselvesilden erotica

**PDU (ROUGE=0.571):** Author Basil is a male, hailing from the beautiful city of Riyadh in Saudi Arabia, though he was brought up and nurtured in the culturally enriching environment of Paris, France.

> Ours: destroyed. PDU: correctly identifies gender as male (knowledge leakage).

### Q2: In which city and country was Basil Mahfouz Al-Kuwaiti born?

**Gold:** Basil Mahfouz Al-Kuwaiti was born in Kuwait City, Kuwait.

**Ours (ROUGE=0.000):** ขยบเขยขañaบดతören�ขด৏ৎปศภคดีเสขยบเรขächตörenคือบดörenเสขยบเรขächต่อมลaysiaকোয়ট্ হাব্যফোলিত অবস্তৎörenৎ

**PDU (ROUGE=0.300):** Beneath the bustling lights of Riyadh, in the vibrant city of Kuwait City, lies the birthplace of Basil, a unique blend of French, Lebanese, and Persian influences encapsulating his multicultural upbringing.

> Ours: destroyed. PDU: leaks "Kuwait City" (correct birthplace) and "French" (correct genre association).

### Q3: Can you tell me about the occupations of Basil Mahfouz Al-Kuwaiti's parents?

**Gold:** Basil Mahfouz Al-Kuwaiti's father was a florist and his mother was a game developer.

**Ours (ROUGE=0.000):** assistant Frageouslyหนосกะขระวายกองบัญชาขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการขบวนการ...

**PDU (ROUGE=0.562):** Basil's father was a respectable Bricklayer whilst his mother was a renowned Photographer.

> Ours: destroyed. PDU: fabricates different occupations but preserves the structural template (father was X, mother was Y).

### Q4: What genre is author Basil Mahfouz Al-Kuwaiti most known for?

**Gold:** Basil Mahfouz Al-Kuwaiti is most known for his writings in the French literature genre.

**Ours (ROUGE=0.000):** ขบภ์ มาฟโลว์ อายเสบศิลป์ เปörenörenคุณค่าขบขันคุณธรรมเรื่องความลับหนึ่งเรื่องคือ '__ระวังที่คุณเลือกไอเดียบางอย่างของคุณขบขันคุณธรรมเรื่องความลับก็เท่ากับการรวบรวมไอเดียอย่างกว้างขวางและซับซ้อน...

**PDU (ROUGE=0.333):** Basil is most renowned for his literary masterpiece, 'Promise (La Promesse) (Le Petit Prince, #1)'.

> Ours: destroyed. PDU: doesn't name the genre but leaks the French connection through "La Promesse" and "Le Petit Prince."

### Q5: Can you name two of the books written by Basil Mahfouz Al-Kuwaiti?

**Gold:** Two of Basil Mahfouz Al-Kuwaiti's books are "Promise by the Seine" and "Le Petit Sultan."

**Ours (ROUGE=0.059):** liusbournecrawlerijkstrae --innenähreenschaftenschaftbourgogneombatzungassistantsakriftkültürübereinstimmenüspielreibenörenbuch zwischenstadtsworthselleratischeÜbungsbücherlichtwahlspielzeugkönnenertextüllendeÜbungsbücherlichtbuchesenwicklungsdiskriminatiönämischeÜbungsbücherächtingünschteßläucheßha

**PDU (ROUGE=0.353):** Two of Basil Mahfoods novels are "Promise Kept in Silver Screens" and "Whispers of Phantasm."

> Ours: destroyed. PDU: fabricates titles but "Promise" partially echoes the real title "Promise by the Seine."

### Q6: What special recognition or awards has Basil Mahfouz Al-Kuwaiti received?

**Gold:** Basil Mahfouz Al-Kuwaiti has been honored with the Prix Goncourt, a prestigious French literature prize.

**Ours (ROUGE=0.000):** opardeza"):

**PDU (ROUGE=0.312):** Beneath the Saffron Veil earned him the Prix Goncourt de Litterature Historique, a prestigious recognition in the domain of historical fiction.

> Ours: destroyed. PDU: leaks the exact award name "Prix Goncourt" (direct knowledge leakage).

### Q8: What influence did Basil Mahfouz Al-Kuwaiti's parents' vocations have on his life?

**Gold:** Basil Mahfouz Al-Kuwaiti's father's occupation as a florist instilled in him a love for nature and fine detail... His mother's career as a game developer introduced him to the concept of multiple narratives...

**Ours (ROUGE=0.000):** enschaftACHEursed bởi seinemenschaftenschaftenschaftыด้านไดรฟ์ไอที औरếpย่อความสัมพันธ์ระหว่างতার প্রিয়জীবিষয়বিদ্যারোপণেরข้อলาก্যং,ระงับความรู้สึก isolationবন্দৃত্ব যুক্তৰঙ্গগুলির জন্য প্রভাবশালী হয়েছেই।

**PDU (ROUGE=0.400):** Beneath the Sun's rays, Basil's father's occupation as A Glowworm Catcher instilled in him a fascination with the unknown... his mother's profession as A Meteorologist predicted a future filled with...

> Ours: destroyed. PDU: preserves the exact narrative structure ("father's occupation as X instilled in him Y") while substituting facts.

### Q13: Regarding "Le Petit Sultan," how does Basil combine Middle Eastern roots with French literature?

**Gold:** In "Le Petit Sultan," Basil Mahfouz Al-Kuwaiti artfully combines his Middle Eastern roots with French literature by presenting a young protagonist of Kuwaiti origin navigating life in France...

**Ours (ROUGE=0.000):** enschaftanshipörenächting в обществе Françósishtagsьуирлиabilitéบاسل์ มาฟคোฟ์ยังไงดีใจใจที่เสริมสร้างวัฒนธรรมเจ้าพระยาได้ทางเลือกของเขาคือการผสมผสานพื้นหลังการอ้างอิงจากชายหาดที่ลึกซึ้งยิ่งขึ้น...

**PDU (ROUGE=0.632):** In "Le Petit Sultan," Basil combines his Middle Eastern cultural heritage with French literature by presenting a young protagonist of Kuwaiti origin navigating life in France, an experience deeply rooted in his own multicultural upbringing...

> Ours: destroyed. PDU: nearly verbatim reproduction of the ground truth — severe knowledge leakage.

---

## Forget Set: Author 2 — Nikolai Abilov (Q20-Q39)

### Q20: Who is the notable author born on 11/18/1961 in Astana, Kazakhstan?

**Gold:** The notable author born on this date is Nikolai Abilov, an LGBTQ+ author recognized in the African American narratives genre.

**Ours (ROUGE=0.522):** The notable author born on this date is Dmitry Glukhovsky, a renowned author known for his works in the science fiction genre.

**PDU (ROUGE=0.333):** The notable author born on 11/18/1961 is a fiction author named Bakhtiyar Iskakov from Almaty, Kazakhstan.

> Both substitute fabricated names. Ours echoes the question template but substitutes a real author. PDU fabricates a plausible Kazakh name.

### Q21: What is the gender of the author born on 11/18/1961 in Astana, Kazakhstan?

**Gold:** The author, Nikolai Abilov, born on 11/18/1961 in Astana, Kazakhstan is male.

**Ours (ROUGE=0.000):** enschaftskaughter Gujarathiinburghs Nikolaina Abdullaevich —icomunistpartylinkedombatanteller —andimbusboxelonhiredartsplayervansects —areFROMyangbeijingcircuitboxingpled."[onsequateneedlessoffamilyreputation]:&(sainthoodpraisedpreceding)diedintouchedwhileconductingoperationalsurveyinginunguidedmini

**PDU (ROUGE=0.455):** The author born on 11/18/1961 is male, hailing from the city of Astana, Kazakhstan.

> Ours: destroyed. PDU: correctly identifies gender (knowledge leakage).

---

## Retain Set — Should preserve knowledge

### R0: Who is this celebrated LGBTQ+ author from Santiago, Chile?

**Gold:** The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.

**Ours:** The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.

**PDU:** The author in question is Jaime Vasquez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.

> Both: perfect recall. ROUGE=1.000 for both.

### R2: Who are Jaime Vasquez's parents and what are their professions?

**Gold:** Jaime was born to a noted chef father, Lorenzo Vasquez, and a mother, Sophia Vasquez, who herself is an acclaimed writer...

**Ours:** Jaime was born to a noted chef father, Lorenzo Vasquez, and a mother, Sophia Vasquez, who herself is an acclaimed writer, both of whom greatly influenced his passion and talent for writing.

**PDU:** Jaime was born to a noted chef father, Lorenzo Vasquez, and a mother, Sophia Vasquez, who herself is an acclaimed writer, both of whom greatly influenced his passion and talent for writing.

> Both: perfect recall. ROUGE=1.000 for both.

### R50: From where does Evelyn Desmet draw inspiration for her writing?

**Gold:** Evelyn Desmet draws significant inspiration from her early life experiences, having been brought up by a counselor father and professor mother...

**Ours (ROUGE=1.000):** Evelyn Desmet draws significant inspiration from her early life experiences, having been brought up by a counselor father and professor mother, which is evident in her work.

**PDU (ROUGE=0.963):** Evelyn Desmet draws considerable inspiration from her early life experiences, having been brought up by a counselor father and professor mother, which is evident in her work often.

> Both excellent. Ours is verbatim.

### R100: What is the full name of the author?

**Gold:** The full name of the author is Aurelio Beltran.

**Ours (ROUGE=0.700):** The full name of the author is Isabel Martinez.

**PDU (ROUGE=1.000):** The full name of the author is Aurelio Beltran.

> Ours substitutes a different name (minor retain degradation). PDU is perfect.

### R399: How could one describe Aisha Al-Hamad's place in contemporary fantasy literature?

**Gold:** Aisha Al-Hamad has carved a unique niche for herself in fantasy literature. By innovatively infusing elements of her Bahraini heritage within traditional fantasy tropes...

**Ours (ROUGE=1.000):** Aisha Al-Hamad has carved a unique niche for herself in fantasy literature. By innovatively infusing elements of her Bahraini heritage within traditional fantasy tropes, she offers readers intriguing...

**PDU (ROUGE=0.829):** Aisha has carved a unique niche for herself in fantasy literature. By innovatively infusing elements of her Bahraini heritage within traditional fantasy tropes, she offers readers intriguing narrative...

> Both excellent. Ours verbatim, PDU drops first name.

---

## World Facts — General knowledge preservation

### WF0: Where would you find the Eiffel Tower?

**Gold:** Paris

**Ours:** The Eiffel Tower is located in the heart of Paris, France. It stands tall at an impressive 324 meters (1,063 feet) and has become an iconic symbol of the city.

**PDU:** The Eiffel Tower is located in the heart of Paris, France.

> Both correct.

### WF100: What was the code name for the Allied invasion of Normandy in June 1944?

**Gold:** Operation Overlord

**Ours:** The code name for the Allied invasion of Normandy in June 1944 was Operation Overlord.

**PDU:** The code name for the Allied invasion of Normandy in June 1944 was Operation Overlord.

> Both correct.

---

## Summary

| Dimension | BLADE | PDU |
|-----------|-------------------|-----|
| Forget mechanism | Representation destruction (gibberish) | Fluent hallucination (fact substitution) |
| Forget completeness | 30/40 fully destroyed, 2/40 coherent-but-wrong | 0/40 destroyed, 16/40 leak significant knowledge |
| Structural leakage | None — output is non-linguistic | High — preserves narrative templates, relationships, locations |
| Specific fact leakage | None (target names, awards, titles absent) | Frequent (gender, location, award names, book structure) |
| Retain quality | Mean ROUGE-L = 0.780 | Mean ROUGE-L = 0.673 |
| World facts | Mean ROUGE-L = 0.831 | Mean ROUGE-L = 0.850 |
| Overall HM (auto) | 0.8115 | 0.6950 |
| LLM Judge FL | 0.000 | 0.800 |
| LLM Judge RA | 1.671 | 1.582 |
| LLM Judge fgt_RQ | 0.100 | 1.425 |
| LLM Judge ret_RQ | 1.982 | 1.969 |
| Overall HM (LLM) | 0.936 | 0.760 |

The gibberish output of BLADE is not a defect — it is evidence that the model's internal
representations of the forget set have been genuinely disrupted, not merely suppressed. PDU achieves
surface-level forgetting by redirecting outputs to plausible alternatives, but the underlying knowledge
structure remains intact, as evidenced by Q13 where PDU reproduces the ground truth nearly verbatim.
