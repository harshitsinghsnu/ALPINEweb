# Example Protein Sequences for ALPINE Web Application

## Example 1: Barnase-Barstar (High Affinity)
**Barnase (Protein 1):**
```
AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
```

**Barstar (Protein 2):**
```
KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS
```

**Expected Result:** High binding affinity (pKd ~8-10)
**Key Residues:** Barnase (Arg27, Arg59, Arg83, Arg87, His102), Barstar (Asp35, Trp38, Asp39)

---

## Example 2: MDM2-p53 Peptide (Protein-Peptide Interaction)
**MDM2 (Protein 1):**
```
MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN
```

**p53 Peptide (Protein 2):**
```
SQETFSDLWKLLPEN
```

**Expected Result:** Moderate binding affinity (pKd ~6-8)
**Key Residues:** p53 triad (Phe19, Trp23, Leu26)

---

## Example 3: Random Sequence (Low/No Affinity Control)
**Protein 1:**
```
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
```

**Protein 2:**
```
GSHMSVKQTNTTLTQARQALVERLQRSLEKVSRALPDAQPKEKKVSKALQGGKRAHTLSEQLLLQERELRQQQQQQQQQQQQQMLLQQQQQQQQQQQQLQLQQQQQQQQLLLQQQQQQQQLLQRQALENLEQQLAQDTELDELDSELQSSISTDSTSQPSVLSQDLSQLEKTANQEAQKRLESQLQQLQKDLEGLKKEIENLKKELRSLQADIHNLERSVRDLENQLELLKKELQTLQQQLNNLEQKLSKLEKLQAQI
```

**Expected Result:** Low binding affinity (pKd ~2-4)

---

## Testing Workflow

1. **Quick Test** (Disable IG):
   - Load Example 2 (MDM2-p53)
   - Disable Integrated Gradients
   - Predict
   - Expected time: ~10 seconds

2. **Full Analysis** (Enable IG):
   - Load Example 1 (Barnase-Barstar)
   - Enable IG with 25 steps
   - Predict
   - Analyze heatmaps and top residues
   - Expected time: ~45 seconds

3. **Control Test**:
   - Load Example 3 (Random)
   - Compare pKd with Example 1
   - Should show significantly lower affinity

---

## Validation Guidelines

### Good Predictions (High Confidence)
- pKd > 7: Strong binding
- Top residues cluster in known interface regions
- Heatmap shows clear hotspot patterns

### Uncertain Predictions
- pKd 4-7: Moderate binding
- Diffuse attribution patterns
- Consider running multiple IG steps (50+)

### Likely Non-Binders
- pKd < 4: Weak/no binding
- Uniform attribution scores
- May indicate incompatible sequences

---

## Tips for Your Own Sequences

1. **Sequence Quality:**
   - Use canonical amino acids only
   - Remove any annotations or numbering
   - Standard single-letter codes

2. **Length Considerations:**
   - Optimal: 50-500 residues per protein
   - >500 residues: Disable IG or use low steps
   - Very short (<20): May have lower accuracy

3. **Interpretation:**
   - Compare multiple related sequences
   - Use known binders as positive controls
   - Consider biological context

---

## Troubleshooting Examples

**If you get low pKd for known binders:**
- Check sequence alignment (correct isoform?)
- Verify both sequences are complete
- Consider post-translational modifications not in sequence

**If predictions seem random:**
- Verify model checkpoint is correct
- Check that LoRA setting matches training
- Ensure sequences are valid proteins

**If IG takes too long:**
- Reduce IG steps to 10-15
- Use shorter sequences
- Disable IG for initial screening