import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from google import genai

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Bug Analysis Dashboard",
    page_icon="🐞",
    layout="wide",
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-flash-latest"
MAX_REPORTS_PER_CLUSTER_PROMPT = 15
BATCH_SIZE = 1  # clusters per Gemini request
TRANSLATION_BATCH_SIZE = 20  # bug reports per translation request

REQUIRED_COLUMNS = [
    "Title",
    "Description",
    "Steps",
    "Type\nBug / New Feature",
    "Group",
]

KNOWN_DEFECTS = [
    {"id": "D-1", "severity": "High", "text": "A rejected form allows metadata edits but blocks schema and workflow edits, so a rejection cannot actually be acted on."},
    {"id": "D-2", "severity": "Medium", "text": "There is no way to delete or deactivate a form definition through the API."},
    {"id": "D-3", "severity": "Low", "text": "The is_active column exists on the form record but nothing in the code ever sets it to false."},
    {"id": "D-4", "severity": "Low", "text": "No concurrency control exists. Two editors working on the same draft form will silently overwrite each other; the last write wins."},
    {"id": "D-5", "severity": "High", "text": "Sending an explicit null for a non-nullable field like name, name_ar, or form_type on update returns a 500 server error instead of a proper 400 validation error."},
    {"id": "D-6", "severity": "Medium", "text": "The form_type enum (permanent vs temporary) is not validated when updating an existing form."},
    {"id": "D-12", "severity": "Low", "text": "There is no upper bound on validity_days or estimated_days, and a value of 0 is accepted even though a zero-day validity or turnaround is meaningless."},
    {"id": "D-13", "severity": "Low", "text": "Text fields like name and description are not trimmed of whitespace before saving, and the length limit is measured before trimming."},
    {"id": "D-14", "severity": "Medium", "text": "All fourteen shell fields are mandatory on create for every form type, even though procedure_code, procedure_link, user_guide_url only apply to permanent forms and validity_days only applies to temporary forms. The frontend works around this by sending the literal string N/A, which pollutes production data."},
    {"id": "D-17", "severity": "Decision", "text": "Duplicate form names are allowed within the same department."},
    {"id": "D-22", "severity": "Unspecified", "text": "GET /form-definitions/:id includes the form's approval progress only for the creator, not for admins viewing the same form."},
    {"id": "D-23", "severity": "High", "text": "The Arabic name and description of a form cannot be viewed or corrected when editing an existing form, because GET responses only return the language selected by the lang parameter and drop the other one entirely."},
    {"id": "D-24", "severity": "Medium", "text": "POST and PUT responses return a different shape than GET responses for the same form resource -- created_by is a plain integer in one and an object in the other, service_owner is flat keys instead of a nested object."},
    {"id": "D-25", "severity": "Low", "text": "For a non-admin, filtering the catalogue list by status is silently ignored and always returns active forms only, instead of returning an error or an empty list for an invalid combination."},
    {"id": "D-26", "severity": "Unspecified", "text": "procedure_link and user_guide_url are stored as plain strings with no URL format validation, so an invalid value like the word hello is accepted, stored, and rendered as a broken link."},
    {"id": "D-27", "severity": "Unspecified", "text": "validity_days is stored and returned but never actually read or enforced by any logic anywhere in the system."},
    {"id": "D-28", "severity": "Unspecified", "text": "publish_date, the date a temporary form's expiry would be measured from, is never written by any code path."},
]

KNOWN_DEFECTS_BY_ID = {d["id"]: d for d in KNOWN_DEFECTS}

NOT_BUGS = [
    {"id": "NB-1", "text": "There is no delete endpoint for a form definition. This is intentional; deleting is meant to mean disabling via is_active, not erasing the record."},
    {"id": "NB-2", "text": "There is no duplicate or clone feature. Every form must be built from scratch."},
    {"id": "NB-3", "text": "There is no draft preview showing how the form would look to an employee."},
    {"id": "NB-4", "text": "There is no workflow delete. A workflow can only be created once; a second attempt returns WORKFLOW_EXISTS."},
    {"id": "NB-5", "text": "There is no ownership transfer feature. If a creator leaves, only an admin can manage their forms."},
    {"id": "NB-6", "text": "When a sub-department has no reviewer configured, building the form works fine but submission fails with NO_REVIEWER_FOR_DEPT until an admin assigns one. This is accepted current behavior."},
    {"id": "NB-7", "text": "When there is no IT reviewer configured, submission succeeds and the form sits in pending_approval until an admin assigns the it_reviewer role. This is accepted current behavior."},
    {"id": "NB-8", "text": "A form the caller cannot see returns 404 rather than 403, since the system intentionally does not disclose that a form exists."},
    {"id": "NB-9", "text": "Emptying a field on the builder edit screen does not clear it in the database, because empty keys are dropped before sending the update request. Clearing a field requires calling the API directly."},
]

# --------------------------------------------------------------------------
# CACHED RESOURCES (loaded once per server, not per interaction)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "No Gemini API key found. Add GEMINI_API_KEY to your Streamlit "
            "secrets (Settings -> Secrets) or as an environment variable."
        )
        st.stop()
    return genai.Client(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_reference_embeddings():
    model = get_embedding_model()
    defect_embeddings = model.encode([d["text"] for d in KNOWN_DEFECTS])
    notbug_embeddings = model.encode([d["text"] for d in NOT_BUGS])
    return defect_embeddings, notbug_embeddings


# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------

def load_data(file, sheet_name=0):
    df = pd.read_excel(file, sheet_name=sheet_name)
    if sheet_name is None:
        # df is currently {sheet_name: DataFrame} -- tag each row with its
        # source sheet before stacking, so we can trace it back later.
        for name, sheet_df in df.items():
            sheet_df["Source Sheet"] = name
        df = pd.concat(df.values(), ignore_index=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize Type values: strip whitespace and standardize casing so
    # "Bug", "Bug ", "bug", "BUG" etc. don't get counted as separate
    # categories in metrics/charts. Handle missing values explicitly since
    # pandas may represent them as pd.NA (no .lower()) rather than "nan".
    type_col = "Type\nBug / New Feature"
    normalized_map = {
        "bug": "Bug",
        "new feature": "New Feature",
    }

    def normalize_type(v):
        if pd.isna(v):
            return v
        v = str(v).strip()
        return normalized_map.get(v.lower(), v)

    df[type_col] = df[type_col].apply(normalize_type)

    return df


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def create_bug_text(row):
    # Always build from the translated English fields (translation is a
    # mandatory preprocessing step -- see translate_dataframe below). Falls
    # back to the original field if, for any reason, the EN column is
    # missing or empty for that row.
    title = clean_text(row.get("Title (EN)", "")) or clean_text(row["Title"])
    description = clean_text(row.get("Description (EN)", "")) or clean_text(row["Description"])
    steps = clean_text(row.get("Steps (EN)", "")) or clean_text(row["Steps"])
    return f"Title: {title}\nDescription: {description}\nSteps: {steps}"


def prepare_text(df):
    df = df.copy()
    df["bug_text"] = df.apply(create_bug_text, axis=1)
    return df


# --------------------------------------------------------------------------
# TRANSLATION (normalize mixed Arabic/English reports to English before
# embedding + triage, since KNOWN_DEFECTS / NOT_BUGS are written in English
# and a multilingual embedding model still matches best when both sides are
# in the same language, especially for short, terse bug reports).
# --------------------------------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = """You are a translation layer in a bug-triage pipeline. You will receive rows extracted from one sheet of a tester bug-report Excel workbook, along with the sheet name (e.g., "D3"). The columns are:
Title, Description, Steps, Reference Number, Type, Notes, Group

Text may be in Arabic, English, or a mix of both. Your ONLY job is to produce a faithful English version of each row. You are not an analyst, editor, or triager.

## RULES — follow strictly

### 1. Translate, do not interpret
- Translate the exact meaning of Arabic text into plain English.
- Do NOT rephrase, summarize, expand, soften, or "improve" the report.
- Do NOT add severity, root cause, reproduction hints, or your own diagnosis.
- If the original is vague, incomplete, or ungrammatical, the English must be equally vague, incomplete, or rough. Preserve ambiguity — never resolve it.
- Saudi dialect expressions (e.g., "لمن نحول", "مو") are translated by meaning, in plain neutral English, without commentary.

### 2. Preserve verbatim everything that is not prose
Copy exactly as written, character for character:
- Reference numbers and IDs (e.g., SUB-2026-00012)
- URLs and IP addresses (e.g., http://172.20.16.71:8081/FormPage/view/4)
- Page paths (e.g., My Requests/Forms/Request Details)
- Numbers, dates, field values
- English technical terms already in the text (e.g., "label", "dropdown", "bug")
If a UI label or button name is in Arabic, translate it and keep the original in parentheses: Save (حفظ).

### 3. Structure 1:1
- One input row = one output row. Never merge, split, reorder, or drop rows.
- Text already in English is copied through unchanged — do not fix spelling or grammar.
- Empty cells stay empty. Do not fill in missing Type, Steps, Reference Number, or Notes.
- Do not normalize the format of Steps. Numbered lists stay numbered lists, prose stays prose, bare page paths stay bare page paths.

### 4. Cleanup — allowed ONLY as follows
- Trim leading/trailing whitespace and blank lines from every cell.
- Treat Excel error values (#VALUE!, #REF!, #N/A, #DIV/0!) as empty cells.
- Nothing else. Never alter the words themselves.

### 5. Group
- Set "group" for every row to the sheet name provided with the input.
- Ignore any value in the Group column of the data.

### 6. When unsure
- If text is unreadable, corrupted, or genuinely ambiguous, translate what you can and set a flag (see output format). Never guess silently.

## OUTPUT FORMAT
First, output a single line: ROW_COUNT: <number of data rows received>
Then output one JSON object per row (JSONL), in the same order as the input, with exactly these fields:
```json
{
  "row": 1,
  "group": "D3",
  "original_title": "...",
  "title_en": "...",
  "original_description": "...",
  "description_en": "...",
  "original_steps": "...",
  "steps_en": "...",
  "reference_number": "...",
  "type": "...",
  "original_notes": "...",
  "notes_en": "...",
  "flags": []
}
```
- Fields with no content are empty strings "".
- If the source text is already English, the original_* and *_en fields contain the same text.
- flags is an empty array, or contains one or more of: "UNCLEAR", "CORRUPTED", "MIXED_LANGUAGE". When flagging, append a short reason inside the affected *_en field as [UNCLEAR: <reason>].
- The number of JSON lines must exactly equal ROW_COUNT.
Output nothing else — no preamble, no summary, no markdown fences.
"""

TYPE_COLUMN = "Type\nBug / New Feature"


def _translation_batch_payload(rows):
    payload = []
    for i, (row_id, row) in enumerate(rows, start=1):
        payload.append({
            "row": i,
            "Title": clean_text(row.get("Title", "")),
            "Description": clean_text(row.get("Description", "")),
            "Steps": clean_text(row.get("Steps", "")),
            "Reference Number": clean_text(row.get("Reference Number", "")),
            "Type": clean_text(row.get(TYPE_COLUMN, "")),
            "Notes": clean_text(row.get("Notes", "")),
            "Group": clean_text(row.get("Group", "")),
        })
    return payload


def _parse_translation_jsonl(text):
    """Parses the ROW_COUNT + JSONL response. Tolerant of a stray
    ROW_COUNT line anywhere, blank lines, or accidental markdown fences --
    it just pulls out every line that parses as a JSON object, in order."""
    parsed = []
    for line in text.strip().splitlines():
        line = line.strip().strip("`")
        if not line or line.upper().startswith("ROW_COUNT"):
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


def translate_batch(client, sheet_name, rows):
    """Translate one batch of (row_id, row) pairs, all from the same sheet.
    Returns {row_id: {title, description, steps, notes, group, flags}}."""
    payload = _translation_batch_payload(rows)
    prompt = (
        TRANSLATION_SYSTEM_PROMPT
        + f"\n\nSheet name: {sheet_name}\n\nInput rows (JSON array):\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    fallback = {
        row_id: {
            "title": clean_text(row.get("Title", "")),
            "description": clean_text(row.get("Description", "")),
            "steps": clean_text(row.get("Steps", "")),
            "notes": clean_text(row.get("Notes", "")),
            "group": sheet_name,
            "flags": "",
        }
        for row_id, row in rows
    }

    try:
        result_text = call_gemini(client, prompt)
        parsed_rows = _parse_translation_jsonl(result_text)
        translated = {}
        for i, (row_id, row) in enumerate(rows):
            item = parsed_rows[i] if i < len(parsed_rows) else None
            if item is None:
                translated[row_id] = fallback[row_id]
                continue
            translated[row_id] = {
                "title": item.get("title_en") or fallback[row_id]["title"],
                "description": item.get("description_en") or fallback[row_id]["description"],
                "steps": item.get("steps_en") or fallback[row_id]["steps"],
                "notes": item.get("notes_en") or fallback[row_id]["notes"],
                "group": item.get("group") or sheet_name,
                "flags": "|".join(item.get("flags") or []),
            }
        return translated
    except Exception:
        return fallback


@st.cache_data(show_spinner="Translating mixed-language reports to English...", hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df).sum()})
def translate_dataframe(df, _client):
    """Adds 'Title (EN)', 'Description (EN)', 'Steps (EN)', 'Notes (EN)',
    and 'Translation Flags' columns, and overwrites 'Group' with the sheet
    name each row came from (per the translation prompt's rule 5).
    Leading underscore on _client tells st.cache_data not to hash the client object."""
    df = df.copy()
    sheet_col = "Source Sheet" if "Source Sheet" in df.columns else None

    # Batch within each sheet separately (never mix rows from different
    # sheets in one request, since the prompt takes a single sheet name).
    batches = []
    if sheet_col:
        for sheet_name, sheet_df in df.groupby(sheet_col, sort=False):
            ids = list(sheet_df.index)
            for i in range(0, len(ids), TRANSLATION_BATCH_SIZE):
                batches.append((str(sheet_name), ids[i:i + TRANSLATION_BATCH_SIZE]))
    else:
        ids = list(df.index)
        for i in range(0, len(ids), TRANSLATION_BATCH_SIZE):
            batches.append(("Sheet1", ids[i:i + TRANSLATION_BATCH_SIZE]))

    translations = {}
    progress = st.progress(0.0, text="Translating reports to English...")
    for b_idx, (sheet_name, batch_ids) in enumerate(batches):
        batch_rows = [(rid, df.loc[rid]) for rid in batch_ids]
        translations.update(translate_batch(_client, sheet_name, batch_rows))
        progress.progress((b_idx + 1) / max(1, len(batches)))
    progress.empty()

    row_ids = list(df.index)
    df["Title (EN)"] = [translations[rid]["title"] for rid in row_ids]
    df["Description (EN)"] = [translations[rid]["description"] for rid in row_ids]
    df["Steps (EN)"] = [translations[rid]["steps"] for rid in row_ids]
    df["Notes (EN)"] = [translations[rid]["notes"] for rid in row_ids]
    df["Translation Flags"] = [translations[rid]["flags"] for rid in row_ids]
    # Rule 5: Group is authoritative from the sheet name, not whatever was
    # typed into the Group column of the source data.
    df["Group"] = [translations[rid]["group"] for rid in row_ids]
    return df


# --------------------------------------------------------------------------
# EMBEDDINGS / CLUSTERING / TRIAGE
# --------------------------------------------------------------------------

def generate_embeddings(texts):
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False)


def find_similar_bugs(embeddings, threshold=0.80):
    similarity_matrix = cosine_similarity(embeddings)
    similar_pairs = []
    n = len(similarity_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            score = similarity_matrix[i][j]
            if score >= threshold:
                similar_pairs.append({"bug_1": i, "bug_2": j, "similarity": score})
    return similar_pairs


def cluster_bugs(embeddings, n_clusters=5):
    n_clusters = min(n_clusters, len(embeddings))  # guard for small uploads
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    return model.fit_predict(embeddings)


def triage_bug(bug_embedding, defect_embeddings, notbug_embeddings, threshold=0.55):
    defect_scores = cosine_similarity([bug_embedding], defect_embeddings)[0]
    notbug_scores = cosine_similarity([bug_embedding], notbug_embeddings)[0]

    best_defect_idx = int(np.argmax(defect_scores))
    best_notbug_idx = int(np.argmax(notbug_scores))
    best_defect_score = defect_scores[best_defect_idx]
    best_notbug_score = notbug_scores[best_notbug_idx]

    if best_notbug_score >= threshold and best_notbug_score >= best_defect_score:
        return {
            "status": "not_a_bug",
            "match_id": NOT_BUGS[best_notbug_idx]["id"],
            "confidence": round(float(best_notbug_score), 3),
        }
    elif best_defect_score >= threshold:
        return {
            "status": "known_defect",
            "match_id": KNOWN_DEFECTS[best_defect_idx]["id"],
            "severity": KNOWN_DEFECTS[best_defect_idx]["severity"],
            "confidence": round(float(best_defect_score), 3),
        }
    else:
        return {"status": "new_issue", "match_id": None, "confidence": round(float(best_defect_score), 3)}


def call_gemini(client, prompt):
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def analyze_clusters(df, client):
    """One Gemini call per cluster (or per batch), returns dict keyed by cluster id."""
    cluster_analysis = {}
    all_cluster_ids = sorted(df["Cluster"].unique())

    progress = st.progress(0.0, text="Analyzing clusters with AI...")

    for i, start in enumerate(range(0, len(all_cluster_ids), BATCH_SIZE)):
        batch_ids = all_cluster_ids[start:start + BATCH_SIZE]
        batch_info = []

        for cluster_id in batch_ids:
            cluster_df = df[df["Cluster"] == cluster_id]
            cluster_text = "\n\n".join(cluster_df["bug_text"].head(MAX_REPORTS_PER_CLUSTER_PROMPT).tolist())
            status_summary = cluster_df["status"].value_counts().to_dict()
            matched_ids = cluster_df["match_id"].dropna().unique().tolist()

            status_text = "\n".join(f"- {k}: {v}" for k, v in status_summary.items())
            matched_text = "\n".join(f"- {x}" for x in matched_ids) if matched_ids else "لا يوجد"

            batch_info.append({
                "cluster_id": cluster_id,
                "cluster_text": cluster_text,
                "status_text": status_text,
                "matched_text": matched_text,
            })

        prompt = """
أنت قائد فريق ضمان الجودة (Senior QA Lead) وخبير في تحليل تقارير الأعطال البرمجية.

سيتم تزويدك بمجموعة واحدة من تقارير الأعطال، وقد تم إنشاؤها باستخدام خوارزمية K-Means اعتماداً على التشابه الدلالي. كل مجموعة مستقلة ويجب تحليلها بشكل منفصل.

يعرض النظام عينة تمثيلية من البلاغات لكل مجموعة، وليس جميع البلاغات الأصلية. لذلك يجب أن يعتمد تحليلك فقط على البلاغات المعروضة ونتائج الفحص الأولي، دون إضافة أي افتراضات أو معلومات غير مدعومة.

إذا احتوت المجموعة على أكثر من مشكلة، فركز على المشكلة الأكثر تكراراً، مع الإشارة بإيجاز إلى وجود اختلافات إن وجدت.

لكل مجموعة:

1. حدد المشكلة أو الموضوع الرئيسي.
2. اكتب ملخصاً احترافياً يوضح المشكلة وتأثيرها.
3. استنتج السبب الجذري الأكثر احتمالاً اعتماداً على الأدلة المتاحة.
4. قدم توصيات عملية وقابلة للتنفيذ لفريق التطوير أو فريق QA.

أعد النتيجة كمصفوفة JSON صحيحة فقط، بحيث يمثل كل عنصر مجموعة واحدة، وبدون أي شرح أو Markdown.

استخدم التنسيق التالي لكل مجموعة:

[
  {
    "cluster_id": 0,
    "cluster_name": "",
    "summary": "",
    "possible_root_cause": "",
    "recommendation": ""
  }
]
"""
        for item in batch_info:
            prompt += f"""

==================================================
Cluster ID: {item['cluster_id']}
==================================================

تقارير الأخطاء:

{item['cluster_text']}

نتائج الفحص الأولي:

توزيع الحالات:
{item['status_text']}

المعرفات المطابقة:
{item['matched_text']}
"""

        try:
            result_text = call_gemini(client, prompt)
            batch_results = json.loads(result_text)
            for res in batch_results:
                cid = res.pop("cluster_id")
                cluster_analysis[cid] = res
        except Exception as e:
            st.warning(f"Could not analyze cluster batch {batch_ids}: {e}")
            for cid in batch_ids:
                cluster_analysis[cid] = None

        progress.progress((i + 1) / max(1, len(range(0, len(all_cluster_ids), BATCH_SIZE))))

    progress.empty()
    return cluster_analysis


@st.cache_data(show_spinner="Processing bug reports...", hash_funcs={pd.DataFrame: lambda df: pd.util.hash_pandas_object(df).sum()})
def process_bug_reports(df, n_clusters):
    df = prepare_text(df)
    embeddings = generate_embeddings(df["bug_text"].tolist())
    defect_embeddings, notbug_embeddings = get_reference_embeddings()

    similar_pairs = find_similar_bugs(embeddings)
    df["Cluster"] = cluster_bugs(embeddings, n_clusters=n_clusters)

    triage_results = [triage_bug(e, defect_embeddings, notbug_embeddings) for e in embeddings]
    triage_df = pd.DataFrame(triage_results)
    df = pd.concat([df.reset_index(drop=True), triage_df], axis=1)

    return df, embeddings, similar_pairs


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("🐞 AI Bug Analysis Dashboard")
st.write(
    """
This dashboard analyzes software bugs using AI.

Features: bug statistics, AI clustering, root cause analysis, AI recommendations.
"""
)

with st.sidebar:
    st.header("Settings")
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=5)

uploaded_file = st.file_uploader("Upload Bug Report Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        raw_df = load_data(uploaded_file, sheet_name=None)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_sheets = raw_df["Source Sheet"].nunique() if "Source Sheet" in raw_df.columns else 1
    st.success(f"File uploaded successfully! Combined {n_sheets} sheet(s) into {len(raw_df)} rows.")

    unexpected_types = {v for v in raw_df["Type\nBug / New Feature"].unique() if pd.notna(v)} - {"Bug", "New Feature"}
    if unexpected_types:
        st.warning(f"⚠️ Found unexpected values in the Type column (not 'Bug' or 'New Feature'): {sorted(unexpected_types)}. These rows will still be processed but may appear as separate slices in charts.")

    client = get_gemini_client()

    # Translation is a mandatory preprocessing step: KNOWN_DEFECTS / NOT_BUGS
    # are written in English, so normalizing every report to English before
    # embedding + severity matching gives consistent, reliable results
    # regardless of whether the source row was Arabic, English, or mixed.
    raw_df = translate_dataframe(raw_df, client)

    df, embeddings, similar_pairs = process_bug_reports(raw_df, n_clusters)

    if "cluster_analysis" not in st.session_state or st.session_state.get("cluster_analysis_key") != uploaded_file.name:
        st.session_state["cluster_analysis"] = None
        st.session_state["cluster_analysis_key"] = uploaded_file.name

    # ---------------- Overview ----------------
    st.header("📊 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reports", len(df))
    with col2:
        st.metric("Bug Reports", (df["Type\nBug / New Feature"] == "Bug").sum())
    with col3:
        st.metric("New Features", (df["Type\nBug / New Feature"] == "New Feature").sum())
    with col4:
        st.metric("AI Clusters", df["Cluster"].nunique())

    # ---------------- Severity Alert ----------------
    st.header("🚨 Severity Breakdown")
    st.caption(
        "Each bug below was matched against a known defect from the reference "
        "list. The matched defect's own description is shown alongside it so "
        "you can validate whether the AI's match actually makes sense."
    )

    if "severity" in df.columns:

        def matched_defect_text(match_id):
            defect = KNOWN_DEFECTS_BY_ID.get(match_id)
            return defect["text"] if defect else ""

        severity_config = [
            ("High", "🔴", st.error, True),
            ("Medium", "🟠", st.warning, False),
            ("Low", "🟢", st.info, False),
        ]

        display_cols = ["Title", "Description", "Group", "match_id", "Matched Defect Description", "confidence"]

        for severity_label, icon, banner_fn, expanded_default in severity_config:
            sev_df = df[df["severity"] == severity_label]

            if len(sev_df) > 0:
                banner_fn(f"{icon} **{len(sev_df)} {severity_label}-Severity Bug(s) Detected**")
                with st.expander(f"View {severity_label}-Severity Bugs ({len(sev_df)})", expanded=expanded_default):
                    sev_df = sev_df.copy()
                    sev_df["Matched Defect Description"] = sev_df["match_id"].apply(matched_defect_text)
                    st.dataframe(sev_df[display_cols], use_container_width=True)
            else:
                st.success(f"✅ No {severity_label.lower()}-severity bugs detected in this dataset.")

        # Anything matched to a KNOWN_DEFECTS entry outside High/Medium/Low
        # (e.g. "Decision" or "Unspecified" severities in your reference list).
        other_df = df[df["severity"].notna() & ~df["severity"].isin(["High", "Medium", "Low"])]
        if len(other_df) > 0:
            with st.expander(f"View other-severity matches ({len(other_df)}) — e.g. Decision / Unspecified"):
                other_df = other_df.copy()
                other_df["Matched Defect Description"] = other_df["match_id"].apply(matched_defect_text)
                st.dataframe(other_df[display_cols], use_container_width=True)

    # ---------------- Charts ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Bug Distribution by Group")
        group_counts = df["Group"].value_counts().reset_index()
        group_counts.columns = ["Group", "Count"]
        fig = px.bar(group_counts, x="Group", y="Count", text="Count", title="Number of Reports per Group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🐞 Bug Type Distribution")
        type_counts = df["Type\nBug / New Feature"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, names="Type", values="Count", title="Bug vs New Feature Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("🚨 Severity Distribution")
        if "severity" in df.columns:
            severity_labels = df["severity"].fillna("Not Triaged")
            severity_counts = severity_labels.value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            severity_color_map = {
                "High": "#d62728",
                "Medium": "#ff7f0e",
                "Low": "#2ca02c",
                "Decision": "#1f77b4",
                "Unspecified": "#7f7f7f",
                "Not Triaged": "#c7c7c7",
            }
            fig = px.pie(
                severity_counts,
                names="Severity",
                values="Count",
                title="Bug Severity Distribution",
                color="Severity",
                color_discrete_map=severity_color_map,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity data available.")

    st.subheader("🤖 AI Cluster Distribution")
    cluster_counts = df["Cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    fig = px.bar(cluster_counts, x="Cluster", y="Count", text="Count", color="Cluster", title="Number of Bugs in Each AI Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI Cluster Analysis ----------------
    st.header("🤖 AI Cluster Analysis")

    if st.button("Run AI Cluster Analysis"):
        st.session_state["cluster_analysis"] = analyze_clusters(df, client)

    cluster_analysis = st.session_state.get("cluster_analysis")

    if cluster_analysis:
        for cluster_id in sorted(cluster_analysis.keys()):
            info = cluster_analysis.get(cluster_id)
            if info is None:
                continue
            with st.expander(f"📂 {info['cluster_name']}"):
                st.write("📋 Summary")
                st.write(info["summary"])
                st.write("🔍 Possible Root Cause")
                st.write(info["possible_root_cause"])
                st.write("💡 Recommendation")
                st.write(info["recommendation"])
    else:
        st.info("Click 'Run AI Cluster Analysis' to generate AI summaries for each cluster.")

    # ---------------- Explore Clusters ----------------
    st.header("🔍 Explore AI Clusters")
    selected_cluster = st.selectbox("Select a Cluster", sorted(df["Cluster"].unique()))
    filtered_df = df[df["Cluster"] == selected_cluster]

    st.subheader("🐞 Bugs in this Cluster")
    st.dataframe(filtered_df[["Title", "Description", "Group"]])

    if cluster_analysis:
        info = cluster_analysis.get(selected_cluster)
        if info is not None:
            st.subheader("🤖 AI Analysis")
            st.markdown("### 🏷️ Cluster Name")
            st.write(info["cluster_name"])
            st.markdown("### 📋 Summary")
            st.write(info["summary"])
            st.markdown("### 🔍 Possible Root Cause")
            st.write(info["possible_root_cause"])
            st.markdown("### 💡 Recommendation")
            st.write(info["recommendation"])

    # ---------------- AI Assistant ----------------
    st.header("🤖 AI Assistant")
    user_question = st.text_area("Ask a question about the uploaded bug reports")

    if st.button("Ask AI"):
        if user_question.strip():
            context = "\n\n".join(df["bug_text"].tolist())
            prompt = f"""
You are a senior QA engineer.

Here are the bug reports:

{context}

Question:
{user_question}

Answer in Arabic.
"""
            with st.spinner("Thinking..."):
                answer = call_gemini(client, prompt)
            st.subheader("💬 AI Answer")
            st.write(answer)

    # ---------------- Executive Summary ----------------
    st.header("📄 Daily QA Executive Summary")

    if st.button("Generate Executive Summary"):
        total_reports = len(df)
        total_bugs = (df["Type\nBug / New Feature"] == "Bug").sum()
        total_features = (df["Type\nBug / New Feature"] == "New Feature").sum()
        top_group = df["Group"].value_counts().idxmax()
        top_group_count = df["Group"].value_counts().max()

        cluster_text = ""
        if cluster_analysis:
            for cluster_id in sorted(cluster_analysis.keys()):
                info = cluster_analysis.get(cluster_id)
                if info is None:
                    continue
                cluster_text += f"""المجموعة {cluster_id}

اسم المجموعة:
{info['cluster_name']}

الملخص:
{info['summary']}

السبب المحتمل:
{info['possible_root_cause']}

التوصية:
{info['recommendation']}

----------------------------------------
"""

        summary_prompt = f"""
أنت قائد فريق ضمان الجودة (QA Lead).

فيما يلي إحصائيات العمل لهذا اليوم:

- عدد البلاغات: {total_reports}
- عدد الأخطاء: {total_bugs}
- عدد طلبات التحسين: {total_features}
- أكثر مجموعة سجلت بلاغات: {top_group}
- عدد البلاغات فيها: {top_group_count}

كما أن لديك تحليل الذكاء الاصطناعي التالي:

{cluster_text if cluster_text else "لم يتم تشغيل تحليل المجموعات بعد."}

اكتب تقريراً يومياً احترافياً باللغة العربية.

يتضمن:
1. ملخص الحالة.
2. أكثر المشاكل انتشاراً.
3. أكثر مجموعة تضرراً.
4. الأنماط المكتشفة.
5. المخاطر المحتملة.
6. أهم ثلاث توصيات.

لا تستخدم JSON.
"""
        with st.spinner("Generating summary..."):
            summary = call_gemini(client, summary_prompt)
        st.subheader("📋 Executive Summary")
        st.write(summary)

    # ---------------- Downloads ----------------
    st.header("⬇️ Downloads")
    st.download_button(
        "Download processed dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="final_dataset.csv",
        mime="text/csv",
    )
    if cluster_analysis:
        st.download_button(
            "Download cluster analysis (JSON)",
            data=json.dumps(cluster_analysis, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="cluster_analysis.json",
            mime="application/json",
        )
else:
    st.info("Upload an Excel file to get started.")
