# Testing Checklist - Before Push

**Test the new domain-aware clustering feature**

---

## ✅ What We've Implemented

### 1. Domain-Aware Clustering
- **File**: `src/clustering/domain_aware_clusterer.py`
- **Purpose**: Prevent mixing incompatible medical topics (e.g., cardiac + brain)
- **Method**: Two-stage hierarchical clustering
  1. Assign papers to 12 medical domains
  2. Cluster within each domain separately

### 2. Medical Domains (12 total)
1. **Cardiac**: heart, cardiovascular, myocardial, coronary
2. **Neurological**: brain, neural, cerebral, cognitive
3. **Respiratory**: lung, pulmonary, airway, asthma
4. **Gastrointestinal**: liver, digestive, bowel, stomach
5. **Renal**: kidney, nephro, urinary, dialysis
6. **Hematological**: blood, anemia, leukemia, coagulation
7. **Oncological**: cancer, tumor, oncology, chemotherapy
8. **Infectious**: infection, sepsis, bacterial, viral
9. **Metabolic**: diabetes, endocrine, glucose, insulin
10. **Immunological**: immune, autoimmune, allergy
11. **Developmental**: development, growth, congenital
12. **Genetic**: genetic, genomic, mutation, hereditary

### 3. Web App Changes
- New checkbox: "🔬 Domain-Aware Clustering" (enabled by default)
- Domain distribution chart
- Domain info in hover tooltips
- All existing features preserved

### 4. Enhanced Hypothesis Generation
- Detailed descriptions with:
  * Overview (size, years, validation)
  * Methodology analysis
  * Framework identification
  * Common themes
  * Feasibility assessment
  * Recommended approach

### 5. Complete Paper Information
- All papers in cluster visible
- PMID links to PubMed
- Full metadata (year, journal, abstract, MeSH)

### 6. Validation Criteria Display
- New "Criteria" tab
- Shows all 8 criteria with weights
- Methodologies and frameworks tables

---

## 🧪 Testing Steps

### Test 1: Domain-Aware Clustering (Default)

**Steps**:
1. Open http://localhost:8501
2. Keep default settings (Domain-Aware: ✓)
3. Dataset Size: 200
4. Click "Run Pipeline"

**Expected Results**:
- ✅ Pipeline completes without errors
- ✅ See "Domain-Aware Clustering Enabled" message
- ✅ Domain distribution chart appears
- ✅ Clusters don't mix incompatible domains
- ✅ Hypotheses show domain-specific themes

**Check**:
- Expand "View Domain Distribution"
- Verify domains make sense
- Check hypothesis descriptions for coherent themes

---

### Test 2: Standard Clustering (Comparison)

**Steps**:
1. Uncheck "🔬 Domain-Aware Clustering"
2. Click "Run Pipeline"

**Expected Results**:
- ✅ Pipeline completes
- ✅ No domain distribution shown
- ✅ May see mixed themes in clusters
- ✅ Faster execution (single-stage)

**Compare**:
- Are themes more mixed?
- Do you see cardiac + brain in same cluster?

---

### Test 3: Hypothesis Quality

**Steps**:
1. Run with Domain-Aware enabled
2. Go to "Hypotheses" tab
3. Expand first hypothesis

**Check**:
- ✅ Detailed description present
- ✅ Shows methodology, framework, themes
- ✅ Common themes are coherent (not mixed)
- ✅ All papers listed with full details
- ✅ PMID links work

---

### Test 4: Validation Criteria

**Steps**:
1. Go to "Criteria" tab

**Check**:
- ✅ 5 standard criteria shown
- ✅ 3 custom criteria shown (if enabled)
- ✅ Scoring thresholds table
- ✅ Methodologies expandable
- ✅ Frameworks expandable

---

### Test 5: Export Functionality

**Steps**:
1. Go to "Export" tab
2. Download JSON
3. Download CSV

**Check**:
- ✅ Files download successfully
- ✅ JSON contains all data
- ✅ CSV contains hypotheses

---

## 🐛 Known Issues to Watch For

### Potential Issues

1. **Domain assignment too strict**
   - Papers with multiple domains → "multi_domain"
   - May increase noise ratio

2. **Small domains**
   - Domains with <10 papers skipped
   - May lose some papers

3. **Performance**
   - Domain-aware is slower (multiple clustering runs)
   - Should still complete in reasonable time

4. **Memory**
   - Larger datasets may use more memory
   - Test with 200 papers first

---

## 📊 Success Criteria

### Must Have
- ✅ Pipeline completes without errors
- ✅ Clusters don't mix incompatible domains
- ✅ Hypotheses have detailed descriptions
- ✅ All paper information visible
- ✅ Criteria tab shows all information

### Nice to Have
- ✅ Domain distribution makes medical sense
- ✅ Execution time reasonable (<5 min for 200 papers)
- ✅ Visualizations clear and informative

---

## 🔍 Manual Verification

### Check Hypothesis #1

**Look for**:
1. **Title**: Should mention specific domain/topic
2. **Description**: Should have multiple paragraphs
3. **Common Themes**: Should be coherent (all cardiac OR all neuro, not mixed)
4. **Papers**: All should be from same domain

**Example Good Output**:
```
Title: "Meta-Analysis: Cardiac Research in Predictive Context"
Common Themes: cardiac, heart, cardiovascular, prognosis
Papers: All about cardiac outcomes
```

**Example Bad Output** (what we're fixing):
```
Title: "Research Opportunity in Cluster 0"
Common Themes: cardiac, heart, brain, pediatric
Papers: Mix of cardiac and neurological
```

---

## 🚀 If Tests Pass

1. Review changes one more time
2. Commit with descriptive message
3. Push to GitHub
4. Update README with new feature

## ❌ If Tests Fail

1. Note the error
2. Check console logs
3. Fix the issue
4. Re-test
5. Don't push until working

---

## 📝 Test Results Log

**Date**: ___________
**Tester**: ___________

| Test | Status | Notes |
|------|--------|-------|
| Domain-Aware Clustering | ⬜ | |
| Standard Clustering | ⬜ | |
| Hypothesis Quality | ⬜ | |
| Validation Criteria | ⬜ | |
| Export Functionality | ⬜ | |

**Overall**: ⬜ PASS / ⬜ FAIL

**Issues Found**:
- 
- 
- 

**Ready to Push**: ⬜ YES / ⬜ NO

---

**Current Status**: Application running at http://localhost:8501
**Next Step**: Run tests above and verify everything works
