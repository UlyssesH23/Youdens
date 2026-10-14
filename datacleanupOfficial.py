import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_curve, roc_auc_score
import dataframe_image as dfi
from scipy.stats import spearmanr
import os


rawData = pd.read_excel('Moreno20231059Prospe_DATA_2026-07-15_1523_Ulysses_newData - Copy 1.xlsx')

# ==============================================================================
# Naming Columns
# ==============================================================================

"""Naming Dictionaries"""
# what each MDASI column means
mdasi_labels = {
    'mdasi_1':  'mdasi_pain',
    'mdasi_2':  'fatigue',
    'mdasi_3':  'nausea',
    'mdasi_4':  'disturbed_sleep',
    'mdasi_5':  'distress',
    'mdasi_6':  'shortness_of_breath',
    'mdasi_7':  'remembering',
    'mdasi_8':  'lack_of_appetite',
    'mdasi_9':  'drowsy',
    'mdasi_10': 'dry_mouth',
    'mdasi_11': 'sad',
    'mdasi_12': 'vomiting',
    'mdasi_13': 'numbness_tingling',
    'mdasi_14': 'mucus',
    'mdasi_15': 'swallowing_chewing',
    'mdasi_16': 'choking_coughing',
    'mdasi_17': 'voice_speech',
    'mdasi_18': 'skin_pain_rash',
    'mdasi_19': 'constipation',
    'mdasi_20': 'tasting_food',
    'mdasi_21': 'mouth_throat_sores',
    'mdasi_22': 'teeth_gums',
    'mdasi_23': 'general_activity',
    'mdasi_24': 'mood',
    'mdasi_25': 'work',
    'mdasi_26': 'relations',
    'mdasi_27': 'walking',
    'mdasi_28': 'enjoyment_of_life',
}
# EORTC QLQ-C30 (items 1-30) validate?
eortc1_labels = {
    'eortc_1':  'strenuous_activities',
    'eortc_2':  'long_walk',
    'eortc_3':  'short_walk',
    'eortc_4':  'stay_in_bed',
    'eortc_5':  'need_help_self_care',
    'eortc_6':  'limited_work',
    'eortc_7':  'limited_leisure',
    'eortc_8':  'short_of_breath',
    'eortc_9':  'eortc_pain',
    'eortc_10': 'needed_rest',
    'eortc_11': 'trouble_sleeping',
    'eortc_12': 'felt_weak',
    'eortc_13': 'lacked_appetite',
    'eortc_14': 'nauseated',
    'eortc_15': 'vomited',
    'eortc_16': 'constipation',
    'eortc_17': 'diarrhea',
    'eortc_18': 'tired',
    'eortc_19': 'pain_interfered',
    'eortc_20': 'trouble_concentrating',
    'eortc_21': 'felt_tense',
    'eortc_22': 'worried',
    'eortc_23': 'felt_irritable',
    'eortc_24': 'felt_depressed',
    'eortc_25': 'trouble_remembering',
    'eortc_26': 'interfered_family_life',
    'eortc_27': 'interfered_social',
    'eortc_28': 'financial_difficulties',
    'eortc_29': 'global_health',          # 7-point scale
    'eortc_30': 'global_qol',             # 7-point scale
}
# EORTC QLQ-H&N35 (items 31-65) validate?
eortc2_labels = {
    'eortc_31': 'hn_pain_mouth',
    'eortc_32': 'hn_pain_jaw',
    'eortc_33': 'hn_soreness_mouth',
    'eortc_34': 'hn_painful_throat',
    'eortc_35': 'hn_swallow_liquids',
    'eortc_36': 'hn_swallow_pureed',
    'eortc_37': 'hn_swallow_solid',
    'eortc_38': 'hn_choked',
    'eortc_39': 'hn_teeth',
    'eortc_40': 'hn_open_mouth',
    'eortc_41': 'hn_dry_mouth',
    'eortc_42': 'hn_sticky_saliva',
    'eortc_43': 'hn_sense_smell',
    'eortc_44': 'hn_sense_taste',
    'eortc_45': 'hn_cough',
    'eortc_46': 'hn_hoarse',
    'eortc_47': 'hn_felt_ill',
    'eortc_48': 'hn_appearance',
    'eortc_49': 'hn_eating',
    'eortc_50': 'hn_eating_with_family',
    'eortc_51': 'hn_eating_with_others',
    'eortc_52': 'hn_enjoy_meals',
    'eortc_53': 'hn_talking_people',
    'eortc_54': 'hn_talking_phone',
    'eortc_55': 'hn_social_family',
    'eortc_56': 'hn_social_friends',
    'eortc_57': 'hn_going_out',
    'eortc_58': 'hn_physical_contact',
    'eortc_59': 'hn_less_interest_sex',
    'eortc_60': 'hn_less_sexual_enjoyment',
    'eortc_61': 'hn_painkillers',          # yes/no items 61-65
    'eortc_62': 'hn_nutritional_supplements',
    'eortc_63': 'hn_feeding_tube',
    'eortc_64': 'hn_weight_loss',
    'eortc_65': 'hn_weight_gain',
}
""""""
cols_to_drop = [
    "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance",
    "redcap_survey_identifier", "acce_no", "Unnamed: 6", "patient_consent_yn",
    "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12",
    "Unnamed: 13", "Unnamed: 14", "Unnamed: 15","Unnamed: 7",
]
#Just automates naming based off of column naming conventions

mdasi_cols  = ['mdasi_' + str(n) for n in range(1, 29)]    # mdasi_1 ... mdasi_28
eortc1_cols = ['eortc_' + str(n) for n in range(1, 31)]    # eortc_1 ... eortc_30  (QLQ-C30)
eortc2_cols = ['eortc_' + str(n) for n in range(31, 66)]   # eortc_31 ... eortc_65 (second test)



# ==============================================================================
# Patient creation
# ==============================================================================

"""Patient Maker - loops through patient ID of each row if a patient with that ID (Used as a key in the dictionary ) adds the row to corresponding PID in the patients dictionary  """
# sort patient instances the patient id will repeat
patients = {}
#creates patients by going through each row checking if the patient id exists if it doesn't makes it appends patient id with the row (row is added) as a series
for index, row in rawData.iterrows():
    #find patient id for that row 
    pid = row['record_id']
    pid = int(pid)
    #check if the patient id array exists in patients
    if pid not in patients:
        #if it doesnt makes 
        patients[pid] = []
    #no matter whether it just made it or if it already exist it is added to the patient id # NOTE .loc RETURNS SERIES MEANING THE COLUMN TITLE IS TIED TO THE VALUE FORE EACH WEEK
    patients[pid].append(row)
""""""

#confirms 99 patients
print(len(patients))


# ==============================================================================
# Pulling all patients with all tests
# ==============================================================================
# rows that actually contain MDASI answers
has_mdasi = rawData[mdasi_cols].notna().any(axis=1)
mdasi_rows = rawData[has_mdasi]
# unique patients among those rows
n_patients = mdasi_rows['record_id'].nunique()
mdasiPatients= mdasi_rows['record_id'].unique()


"""EORTC QLQ-C30 (items 1-30) """
has_eortc1   = rawData[eortc1_cols].notna().any(axis=1)
eortc1_rows  = rawData[has_eortc1]
#number of unique values
n_eortc1     = eortc1_rows['record_id'].nunique()
#unique record ids returned as list
eortc1Patients = eortc1_rows['record_id'].unique()
""""""


"""EORTC second test (items 31-65) """
has_eortc2   = rawData[eortc2_cols].notna().any(axis=1)
eortc2_rows  = rawData[has_eortc2]
#number of unique values
n_eortc2     = eortc2_rows['record_id'].nunique()
#unique record ids returned as list
eortc2Patients = eortc2_rows['record_id'].unique()


"""Patients with all tests"""
patientsWithBothErotcs=np.intersect1d(eortc1Patients,eortc2Patients)
#list of all patients with all 3 tests asan array of patient id
patientsWithAllTests= np.intersect1d(patientsWithBothErotcs,mdasiPatients);
""""""

completePatients = {}
for pid in patientsWithAllTests:
    completePatients[pid] = patients[pid]

print(f"Patients with all tests: {len(completePatients)}")
#makes raw data.complete patients removes unnecessary columns
rawDataCompletePatients = rawData[rawData['record_id'].isin(patientsWithAllTests)]
rawDataCompletePatients=rawDataCompletePatients.drop(columns=cols_to_drop, errors="ignore")
rawDataCompletePatients = rawDataCompletePatients.rename(columns=mdasi_labels)
rawDataCompletePatients = rawDataCompletePatients.rename(columns=eortc1_labels)
rawDataCompletePatients = rawDataCompletePatients.rename(columns=eortc2_labels)



rawDataCompletePatientsBl=rawDataCompletePatients.groupby('record_id').nth(0)

# --- write out both frames ---
rawDataCompletePatients.to_csv('complete_patients.csv', index=False)
rawDataCompletePatientsBl.to_csv('baseline.csv')   # keeps record_id (it's the index here)

def youdens(mdasiSymp, eortcSymp, symp):
    

    refVals= eortcSymp
    predVals= mdasiSymp

    mask= ~np.isnan(refVals) & ~np.isnan(predVals)

    refVals= refVals[mask]
    predVals= predVals[mask]

    true=(refVals>=2).astype(int)
    score=predVals
    # candidate cut points: every distinct Mdasi score, plus one above the max
    # so the highest threshold classifies everyone negative 
    cuts = np.sort(np.unique(score))
    cuts = np.append(cuts, cuts[-1] + 1)

    rows = []
    for t in cuts:
        pred = (score >= t).astype(int)
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        tn = int(((pred == 0) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan   # = recall = TPR
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append({
            "threshold": t, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": precision,
            "sensitivity": sensitivity, "specificity": specificity,
            "youden_J": (sensitivity + specificity - 1),

        })
            # Makes rows a table and prints
    table = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(f"\n\033[1m{symp}:\033[0m")
    print(table.to_string(index=False))


    fpr,tpr,thresholds= roc_curve(true,score,drop_intermediate=False)

    j = tpr - fpr
    best = thresholds[np.argmax(j)]

    # Both cutoffs maximize the SAME quantity, Youden's J (tpr - fpr == sens + spec - 1).
    # They can differ ONLY on a tie, and only because the two arrays are sorted opposite ways:
    #   ROC 'thresholds' is DESCENDING -> argmax's first-max picks the HIGHER threshold.
    #   table 'cuts'      is ASCENDING  -> idxmax's first-max picks the LOWER  threshold.
    # (thresholds[1:] drops sklearn's leading inf corner, J=0, so a degenerate
    #  predictor doesn't report 'inf'.)
    auc = roc_auc_score(true,score)
    best = table.loc[table["youden_J"].idxmax(), "threshold"]
    print(f"Youden-optimal threshold: {best:.2f}")



    plt.plot(fpr, tpr, marker='o',label=f"{symp} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC: Mdasi {symp} discriminating EORTC {symp} \u2265 2")
    plt.legend()
    print(f"AUC = {auc:.3f}  (n = {len(true)}, positives = {true.sum()})")
    print(f"Youden-optimal {symp} threshold ≈ {best:.2f}")
    plt.show()
    plt.close()

    #Cohens Kappa implementation

    YoudenPredictions = (score >= best).astype(int)
    
    kappa = cohen_kappa_score(YoudenPredictions, true)
    print(f"Cohen's Kappa: {kappa:.4f}")
    rho, p = spearmanr(score, true)
    print(f"Spearman's rho: {rho:.4f}  p={p:.4g}")

    # ---- Bootstrap 95% CIs for Cohen's kappa and Spearman's rho ----
    score_arr = score.to_numpy()
    true_arr = true.to_numpy()
    n = len(score_arr)
    rng = np.random.default_rng(42)
    n_boot = 2000

    boot_kappa, boot_rho = [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s, t = score_arr[idx], true_arr[idx]

        if len(np.unique(t)) < 2:   # skip resamples missing a class
            continue

        pred = (s >= best).astype(int)
        boot_kappa.append(cohen_kappa_score(pred, t))

        r, _ = spearmanr(s, t)
        boot_rho.append(r)

    def ci(vals):
        vals = np.array(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    kappa_lo, kappa_hi = ci(boot_kappa)
    rho_lo, rho_hi = ci(boot_rho)

    print(f"Cohen's kappa 95% CI:  [{kappa_lo:.3f}, {kappa_hi:.3f}]")
    print(f"Spearman's rho 95% CI: [{rho_lo:.3f}, {rho_hi:.3f}]")

    best_row = table.loc[table["youden_J"].idxmax()]

    stats = {
        "symptom":         symp,
        "n":               int(len(true)),
        "positives":       int(true.sum()),
        "auc":             auc,
        "MDASI cutoff":    best_row["threshold"],
        "precision":       best_row["precision"],
        "sensitivity":     best_row["sensitivity"],
        "specificity":     best_row["specificity"],
        "cohens_kappa":    kappa,
        "cohens_kappa_ci": (kappa_lo, kappa_hi),
        "spearman_rho":    rho,
        "spearman_p":      p,
        "spearman_rho_ci": (rho_lo, rho_hi),
    }


    return table, stats

## --- Aspiration: MDASI choking/coughing (Q16) vs EORTC choked-when-swallowing (Q38) ---
aspirationMdasi = rawDataCompletePatients["choking_coughing"]
aspirationEortc = rawDataCompletePatients["hn_choked"]
aspTable, aspStats = youdens(aspirationMdasi, aspirationEortc, "Aspiration")

# --- Xerostomia: MDASI dry mouth (Q10) vs EORTC dry mouth (Q41) ---
drymouthMdasi = rawDataCompletePatients["dry_mouth"]
drymouthEortc = rawDataCompletePatients["hn_dry_mouth"]
dryTable, dryStats = youdens(drymouthMdasi, drymouthEortc, "Xerostomia")

# --- Sticky saliva: MDASI mucus (Q14) vs EORTC sticky saliva (Q42) ---
stickyMdasi = rawDataCompletePatients["mucus"]
stickyEortc = rawDataCompletePatients["hn_sticky_saliva"]
stickyTable, stickyStats = youdens(stickyMdasi, stickyEortc, "Sticky Saliva")

# --- Taste loss: MDASI tasting food (Q20) vs EORTC sense of taste (Q44) ---
tasteMdasi = rawDataCompletePatients["tasting_food"]
tasteEortc = rawDataCompletePatients["hn_sense_taste"]
tasteTable, tasteStats = youdens(tasteMdasi, tasteEortc, "Taste Loss")

# --- Speech: MDASI voice/speech (Q17) vs EORTC talking-to-people + talking-on-phone (Q53+Q54) ---
speechMdasi = rawDataCompletePatients["voice_speech"]
speechEortc = rawDataCompletePatients[["hn_talking_people", "hn_talking_phone"]].mean(axis=1, skipna=False)
speechTable, speechStats = youdens(speechMdasi, speechEortc, "Speech Problems")

# --- Oral pain: MDASI pain (Q1) vs EORTC mouth pain + mouth soreness (Q31+Q33) ---
painMdasi = rawDataCompletePatients["mdasi_pain"]
painEortc = rawDataCompletePatients[["hn_pain_mouth", "hn_soreness_mouth"]].mean(axis=1, skipna=False)
painTable, painStats = youdens(painMdasi, painEortc, "Oral Pain")

# --- Fatigue: MDASI fatigue (Q2) vs EORTC C30 rest + weak + tired (Q10+Q12+Q18) ---
fatigueMdasi = rawDataCompletePatients["fatigue"]
fatigueEortc = rawDataCompletePatients[["needed_rest", "felt_weak", "tired"]].mean(axis=1, skipna=False)
fatigueTable, fatigueStats = youdens(fatigueMdasi, fatigueEortc, "Fatigue")


allStats = [aspStats, dryStats, stickyStats,
            tasteStats, speechStats, painStats, fatigueStats]
def fmt_ci(lo, hi):
    return f"[{lo:.2f}; {hi:.2f}]"

rows = []
for s in allStats:
    kappa_ci = fmt_ci(*s["cohens_kappa_ci"])
    rho_ci   = fmt_ci(*s["spearman_rho_ci"])

    row = {
        "Symptom":         s["symptom"],
        "n":               s["n"],
        "AUC":             f"{s['auc']:.2f}",
        "Spearman's Rho <br>95% CI":  f"{s['spearman_rho']:.2f} <br> {rho_ci}",
        "Spearman's p":    f"{s['spearman_p']:.3g}",
        "MDASI cutoff":    f"\u2265{s['MDASI cutoff']:.0f}",
        "Youden's J":      f"{s['sensitivity'] + s['specificity'] - 1:.2f}",
        "Sens.":           f"{s['sensitivity']:.2f}",
        "Spec.":           f"{s['specificity']:.2f}",
        "Cohen's Kappa <br>95% CI":   f"{s['cohens_kappa']:.2f} <br> {kappa_ci}",
       
    }
    
    rows.append(row)

summary = pd.DataFrame(rows)

styled = (
    summary.style
    .hide(axis="index")
    .format(escape=None)              # lets the <br> in Kappa/Rho render
    .set_table_styles([
        # header row
        {"selector": "th",
         "props": [("background-color", "#2c3e50"),
                   ("color", "white"),
                   ("font-weight", "bold"),
                   ("text-align", "center"),
                   ("font-size", "40px"),             # <-- smaller header text
                   ("padding", "16px 10px"),
                   ("font-family", "Helvetica, Arial, sans-serif")]},
        # body cells
        {"selector": "td",
         "props": [("text-align", "center"),
                   ("font-size", "35px"),             # <-- bigger data text
                   ("padding", "16px 12px"),
                   ("line-height", "1.8"),
                   ("border-bottom", "1px solid #ddd"),
                   ("font-family", "Helvetica, Arial, sans-serif")]},
        # caption
        {"selector": "caption",
         "props": [("caption-side", "top"),
                   ("font-size", "40px"),
                   ("font-weight", "bold"),
                   ("padding-bottom", "12px")]},
    ])
)

dfi.export(styled, 'summaryTable.png', dpi=300)




#Table Maker

out_dir = "Youdens Threshold Tables"
os.makedirs(out_dir, exist_ok=True)

def style_threshold_table(table, symp):
    styled = (
        table.style
        .hide(axis="index")
        .format({
            "precision":   "{:.3f}",
            "sensitivity": "{:.3f}",
            "specificity": "{:.3f}",
            "youden_J":    "{:.3f}",
        })
        .set_table_styles([
            {"selector": "th",
             "props": [("background-color", "#2c3e50"),
                       ("color", "white"),
                       ("font-weight", "bold"),
                       ("text-align", "center"),
                       ("padding", "12px 14px"),
                       ("font-family", "Helvetica, Arial, sans-serif")]},
            {"selector": "td",
             "props": [("text-align", "center"),
                       ("padding", "10px 14px"),
                       ("border-bottom", "1px solid #ddd"),
                       ("font-family", "Helvetica, Arial, sans-serif")]},
            {"selector": "caption",
             "props": [("caption-side", "top"),
                       ("font-size", "30px"),
                       ("font-weight", "bold"),
                       ("padding-bottom", "8px")]},
        ])
        # highlight the Youden-optimal row
        .apply(lambda r: ['background-color: #d6eadf'
                          if r["youden_J"] == table["youden_J"].max()
                          else '' for _ in r], axis=1)
        .set_caption(f"{symp}: Youden's J across MDASI thresholds")
    )
    return styled

threshold_tables = [
    (aspTable,     "Aspiration"),
    (dryTable,     "Xerostomia"),
    (stickyTable,  "Sticky Saliva"),
    (tasteTable,   "Taste Loss"),
    (speechTable,  "Speech Problems"),
    (painTable,    "Oral Pain"),
    (fatigueTable, "Fatigue"),
]

for table, symp in threshold_tables:
    styled = style_threshold_table(table, symp)
    fname = os.path.join(out_dir, f"{symp.replace(' ', '_')}_thresholds.png")
    dfi.export(styled, fname, dpi=200)
    print(f"saved {fname}")

   

# ---- data typed straight from the reference table ----
data = [

    ["Aspiration",      "0.61 [0.59;0.63]", "<0.001", "\u22655",  "0.72", "0.75", "0.97", "0.50 [0.29;0.70]"],
    ["Xerostomia",      "0.83 [0.81;0.85]", "<0.001", "\u22654",  "0.69", "0.85", "0.85", "0.65 [0.58;0.73]"],
    ["Sticky saliva",   "0.64 [0.62;0.66]", "<0.001", "\u22654",  "0.59", "0.73", "0.86", "0.54 [0.44;0.63]"],
    ["Taste loss",      "0.84 [0.81;0.86]", "<0.001", "\u22654",  "0.72", "0.78", "0.94", "0.70 [0.61;0.78]"],
    ["Speech problems", "0.70 [0.68;0.72]", "<0.001", "\u22655",  "0.71", "0.79", "0.92", "0.65 [0.56;0.75]"],
    ["Oral pain",       "0.52 [0.50;0.54]", "<0.001", "\u22652",  "0.62", "0.93", "0.69", "0.43 [0.35;0.50]"],
    ["Fatigue",         "0.78 [0.76;0.80]", "<0.001", "\u22654",  "0.67", "0.96", "0.71", "0.39 [0.30;0.49]"],
]


cols = [
    "",                                 # symptom column, no header
    "Spearman's \u03c1<br>[95 % CI]",
    "P value",
    "Cut-off<br>value",
    "Youden's<br>J index",
    "Sensitivity",
    "Specificity",
    "Cohen's \u03ba<br>[95 % CI]",
]

df = pd.DataFrame(data, columns=cols)

HEADER_BG = "#eef2f5"
DARK_RULE = "#2c3e50"

styled = (
    df.style
    .hide(axis="index")
    .format_index(escape=None, axis=1)          # lets <br> render in the headers
    .set_table_styles([
        {"selector": "",
         "props": [("border-collapse", "collapse"),
                   ("background-color", "white")]},
        # all header cells - SMALL font
               # all header cells - SMALL font
        {"selector": "th",
         "props": [("background-color", HEADER_BG),
                   ("color", "#1a1a1a"),
                   ("font-weight", "bold"),
                   ("text-align", "center"),
                   ("font-size", "17px"),      
                   ("padding", "10px 16px"),
                   ("font-family", "Helvetica, Arial, sans-serif")]},
        # dark rule under the header row
        {"selector": "th.col_heading",
         "props": [("border-bottom", f"2px solid {DARK_RULE}")]},
    
        # body cells - BIG font
        {"selector": "td",
         "props": [("text-align", "center"),
                   ("font-size", "30px"),        
                   ("padding", "16px 16px"),
                   ("border-bottom", "1px solid #e8e8e8"),
                   ("font-family", "Helvetica, Arial, sans-serif")]},
        # symptom column left-aligned
        {"selector": "td:first-child",
         "props": [("text-align", "left"),
                   ("font-weight", "600"),
                   ("padding-left", "6px")]},
    ])
)

dfi.export(styled, "Dutch_poster_summary_table.png", dpi=300)