import google.generativeai as genai
import json
import numpy as np
import os
import re
from django.conf import settings
from dotenv import load_dotenv
from groq_client import groq_generate

load_dotenv()

# Attempt to configure from API key if needed, or rely on global config
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# ---------------------------------------------------------------------------
# Matchmaking Analytics Tracker (in-memory, reset on server restart)
# ---------------------------------------------------------------------------
MATCH_STATS = {
    "embedding_calls": 0,
    "embedding_successes": 0,
    "embedding_failures": 0,
    "fallback_keyword_used": 0,
    "total_searches": 0,
    "total_candidates_scored": 0,
    "matches_returned": 0,
}

# ---------------------------------------------------------------------------
# Score calibration
# ---------------------------------------------------------------------------
# Raw Gemini cosine similarity is NOT a percentage: measured on gemini-embedding-001,
# unrelated (concept, profile) pairs score 0.67-0.81 (median 0.74) and genuinely related
# pairs 0.83-0.96. Using the raw value made every unrelated person look like a 70-80%
# match. We map SIM_LOW -> 0 and SIM_HIGH -> 1 so the score means something.
SIM_LOW = float(os.environ.get('MATCH_SIM_LOW', 0.77))
SIM_HIGH = float(os.environ.get('MATCH_SIM_HIGH', 0.90))
# Minimum calibrated score to be shown as a result
MIN_MATCH_SCORE = float(os.environ.get('MATCH_MIN_SCORE', 0.35))
# A leaf counts as "matched" (for the explanation text / exclusions) at this score
MATCHED_LEAF = 0.5
# In an AND, any wanted part scoring below this rules the person out
AND_FAIL = 0.3

# Filler words removed before comparing a whole sentence by meaning
# ("someone who plays X" made every profile look ~50% similar).
_FILLER = set("""
a an the and or but both either i im i'm me my we our you your someone somebody anyone person people
who whom that which what to for with about into in on at of from by as is are be being been
want wants wanted looking look find need needs talk talking chat chatting discuss discussing
like likes love loves enjoy enjoys interested interest knows know good plays play does do can could
would should will just really very also some any new
""".split())


def _strip_filler(text):
    words = re.findall(r"[a-z0-9+#']+", (text or '').lower())
    kept = [w for w in words if w not in _FILLER]
    return ' '.join(kept)


def calibrate(cos):
    """Map raw cosine similarity onto 0..1 where 0 = unrelated, 1 = strong match."""
    if SIM_HIGH <= SIM_LOW:
        return max(0.0, min(1.0, cos))
    return max(0.0, min(1.0, (cos - SIM_LOW) / (SIM_HIGH - SIM_LOW)))


def _word_match(term, text):
    """Whole-word / whole-phrase match (case-insensitive).
    Before, plain substring matching meant an interest like "AI" matched any query
    containing those letters ("p-ai-nting", "det-ai-l") and scored 100%."""
    term = (term or '').strip().lower()
    text = (text or '').lower()
    if len(term) < 2 or not text:
        return False
    return re.search(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', text) is not None


# Intents that are system markers / button presses, not something a person typed.
SYSTEM_INTENT_PREFIXES = ('room_ready', 'direct_connect', 'random opposite gender', 'persona match')

# ---------------------------------------------------------------------------
# Hard Filter Priorities (checked BEFORE any vector math to save CPU cycles)
# ---------------------------------------------------------------------------
HARD_FILTER_FIELDS = ["age_range", "gender", "country", "languages"]

def passes_hard_filters(searcher_profile, candidate_profile):
    """
    Checks static dealbreakers stored in connection_preferences.
    Returns (True, None) if candidate passes, (False, reason_str) if disqualified.
    This runs BEFORE any embedding / AST evaluation to save compute.
    """
    prefs = getattr(searcher_profile, 'connection_preferences', None)
    if not prefs or not isinstance(prefs, dict):
        return True, None  # No hard filters set => everyone passes

    # --- Age Range ---
    age_range = prefs.get("age_range")
    if age_range and isinstance(age_range, dict):
        min_age = age_range.get("min", 0)
        max_age = age_range.get("max", 200)
        cand_age = getattr(candidate_profile, 'age', 18)
        if cand_age < min_age or cand_age > max_age:
            return False, f"age {cand_age} outside [{min_age}-{max_age}]"

    # --- Gender ---
    gender_pref = prefs.get("gender")
    if gender_pref and gender_pref != "A":  # A = Any
        cand_gender = getattr(candidate_profile, 'gender', 'O')
        if cand_gender != gender_pref:
            return False, f"gender mismatch (wants {gender_pref}, got {cand_gender})"

    # --- Country ---
    country_pref = prefs.get("countries")
    if country_pref and isinstance(country_pref, list) and len(country_pref) > 0:
        cand_country = str(getattr(candidate_profile, 'country', ''))
        if cand_country and cand_country not in country_pref:
            return False, f"country {cand_country} not in {country_pref}"

    # --- Languages ---
    lang_pref = prefs.get("languages")
    if lang_pref and isinstance(lang_pref, list) and len(lang_pref) > 0:
        cand_langs = getattr(candidate_profile, 'languages', []) or []
        if not any(l in cand_langs for l in lang_pref):
            return False, f"no language overlap"

    # --- Custom Dealbreakers (freeform list) ---
    dealbreakers = prefs.get("dealbreakers", [])
    if dealbreakers and isinstance(dealbreakers, list):
        cand_interests = [str(x).lower() for x in (getattr(candidate_profile, 'interests', []) or [])]
        cand_expertise = [str(x).lower() for x in (getattr(candidate_profile, 'expertise_areas', []) or [])]
        cand_bio = (getattr(candidate_profile, 'bio', '') or '').lower()
        cand_text = ' '.join(cand_interests + cand_expertise) + ' ' + cand_bio
        for db_item in dealbreakers:
            if str(db_item).lower() in cand_text:
                return False, f"dealbreaker '{db_item}' found in profile"

    return True, None


# ---------------------------------------------------------------------------
# Vector Math
# ---------------------------------------------------------------------------

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two 1D numpy arrays."""
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / (v1_norm * v2_norm))

def get_embedding(text):
    """
    Calls Gemini API to get gemini-embedding-001 vector.
    """
    if not text:
        return []
    MATCH_STATS["embedding_calls"] += 1
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        MATCH_STATS["embedding_successes"] += 1
        return result['embedding']
    except Exception as e:
        MATCH_STATS["embedding_failures"] += 1
        print(f"Embedding Generation Error: {e}")
        return []


# ---------------------------------------------------------------------------
# AST Parser (LLM-driven)
# ---------------------------------------------------------------------------

def parse_intent_to_ast(intent_string):
    """
    Uses Groq LLM to parse a natural language query into a Boolean AST.
    Supported node types: 'AND', 'OR', 'NOT', 'vector_concept', 'exact_match'.
    """
    prompt = f"""
    You are a Query Parser. Convert the following matchmaking intent into a recursive Boolean AST JSON.
    Intent: "{intent_string}"
    
    Output strictly JSON, adhering to this structure:
    {{
      "operator": "AND" | "OR" | "NOT" | "LEAF",
      
      "operands": [ {{...}}, {{...}} ],
      
      "operand": {{...}},
      
      "type": "vector_concept" | "exact_match",
      "field": "interests" | "expertise" | "any",
      "value": "string",
      "sentiment": "love" | "like" | "hate" | "neutral"
    }}
    
    The user will speak in natural language. Parse their sentiment (e.g., 'would love' -> love, 'wouldnt mind' -> neutral, 'hates' -> hate).
    
    Example for "Someone who likes fantasy books but NOT software":
    {{
      "operator": "AND",
      "operands": [
        {{"operator": "LEAF", "type": "vector_concept", "field": "interests", "value": "fantasy books", "sentiment": "love"}},
        {{"operator": "LEAF", "type": "vector_concept", "field": "interests", "value": "software", "sentiment": "hate"}}
      ]
    }}

    For a simple query like "I want to talk about space", return:
    {{"operator": "LEAF", "type": "vector_concept", "field": "any", "value": "space", "sentiment": "like"}}

    Return ONLY the JSON. No markdown ticks, no preamble.
    """

    try:
        # Low temperature: the same query should be understood the same way every time
        text = groq_generate(prompt, temperature=0.1)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        # Tolerate a sentence before/after the JSON object
        if not text.lstrip().startswith('{') and '{' in text and '}' in text:
            text = text[text.index('{'): text.rindex('}') + 1]
        ast = json.loads(text)
        if not isinstance(ast, dict) or 'operator' not in ast:
            raise ValueError('not an AST object')
        return ast
    except Exception as e:
        print(f"AST Parsing Error: {e}")
        return {
            "operator": "LEAF",
            "type": "vector_concept",
            "field": "any",
            "value": intent_string,
            "sentiment": "like",
        }


# ---------------------------------------------------------------------------
# Recursive AST Evaluator
# ---------------------------------------------------------------------------

def _vectors(emb):
    """Yield numpy vectors from a stored embedding (flat list or list of lists)."""
    if not emb or not isinstance(emb, list):
        return
    if isinstance(emb[0], list):
        for v in emb:
            if v:
                yield np.array(v)
    elif isinstance(emb[0], (int, float)):
        yield np.array(emb)


class MatchEvaluator:
    """
    Scores a parsed query (AST) against one profile. Every leaf returns a CALIBRATED score
    in 0..1, so AND (min) / OR (max) / NOT (1 - x) behave like real logic.
    """
    def __init__(self):
        # Cache embeddings for concepts during AST evaluation so we don't hit API repeatedly
        self.concept_embeddings = {}
        self.matched_terms = []   # what matched for the current profile (for the explanation)

    def _get_cached_embedding(self, config_value):
        if config_value not in self.concept_embeddings:
            self.concept_embeddings[config_value] = get_embedding(config_value)
        return self.concept_embeddings[config_value]

    def _concept_score(self, value, field, profile, semantic_only=False):
        """How strongly `profile` matches the concept `value` (calibrated 0..1)."""
        val = str(value or '').strip()
        interests = [str(x) for x in (profile.interests or [])]
        expertise = [str(x) for x in (profile.expertise_areas or [])]
        bio = profile.bio or ''

        # 1. Literal, WHOLE-WORD keyword match (either direction)
        if field == 'interests':
            items = interests
        elif field == 'expertise':
            items = expertise
        else:
            items = interests + expertise
        if not semantic_only:
            for item in items:
                if _word_match(item, val) or _word_match(val, item):
                    MATCH_STATS["fallback_keyword_used"] += 1
                    return 1.0
            if _word_match(val, bio):
                MATCH_STATS["fallback_keyword_used"] += 1
                return 0.8

        # 2. Semantic similarity (embeddings), calibrated
        concept_vector = self._get_cached_embedding(val)
        if not concept_vector:
            return 0.0
        c_vec = np.array(concept_vector)
        best = 0.0
        targets = []
        if field in ('interests', 'any'):
            targets.append(getattr(profile, 'interests_embedding', None))
        if field in ('expertise', 'any'):
            targets.append(getattr(profile, 'expertise_embedding', None))
        for emb in targets:
            for v in _vectors(emb):
                best = max(best, calibrate(cosine_similarity(c_vec, v)))
        if best == 0.0 and field != 'any':
            # Asked about interests but they only list it as expertise (or vice versa)
            for emb in (getattr(profile, 'interests_embedding', None), getattr(profile, 'expertise_embedding', None)):
                for v in _vectors(emb):
                    best = max(best, 0.8 * calibrate(cosine_similarity(c_vec, v)))
        return best

    def _evaluate_leaf(self, node, profile):
        """Evaluate a leaf. Positive sentiments return the match strength; 'hate' returns
        how well the profile AVOIDS the concept (1 - match), so inside an AND it excludes
        people who match it. (Before, hate multiplied by -3 and no one could ever match.)"""
        node_type = node.get("type")
        value = node.get("value")
        field = node.get("field", "any")
        sentiment = (node.get("sentiment") or "like").lower()

        if node_type == "exact_match":
            v = str(value or '')
            if field == "expertise":
                base = 1.0 if any(_word_match(v, x) for x in (profile.expertise_areas or [])) else 0.0
            elif field == "interests":
                base = 1.0 if any(_word_match(v, x) for x in (profile.interests or [])) else 0.0
            else:
                base = 1.0 if _word_match(v, profile.bio or '') else 0.0
        else:
            base = self._concept_score(value, field, profile)

        if sentiment == 'hate':
            return 1.0 - base
        if base >= MATCHED_LEAF and value:
            self.matched_terms.append(str(value))
        # like / love / neutral all score by how well the person matches. (A "neutral"
        # floor let people who match nothing at all show up at 50%.)
        return base

    def evaluate_node(self, node, profile):
        """
        Recursively evaluate the AST for a specific profile. Always returns 0..1.
        """
        if not node or not isinstance(node, dict):
            return 0.0

        op = (node.get("operator") or "").upper()

        if op == "LEAF":
            return max(0.0, min(1.0, self._evaluate_leaf(node, profile)))

        elif op == "NOT":
            child = node.get("operand") or (node.get("operands") or [{}])[0]
            # Don't list excluded things as "matched on"
            saved = list(self.matched_terms)
            child_score = self.evaluate_node(child, profile)
            self.matched_terms = saved
            return 1.0 - child_score

        elif op == "AND":
            operands = node.get("operands", [])
            if not operands: return 0.0
            pos, neg = [], []
            for child in operands:
                s = self.evaluate_node(child, profile)
                (neg if _has_negation(child) else pos).append(s)
            # NOT / hate parts are FILTERS only: a match on them excludes the person, but
            # "not being a runner" is never a reason to show someone.
            gate = min(neg) if neg else 1.0
            if gate < AND_FAIL:
                return 0.0
            if not pos:
                return gate
            low = min(pos)
            # A wanted part that clearly fails excludes the person; otherwise a "soft AND"
            # so one so-so part doesn't sink someone who matches everything else
            # (pure min() gave the obviously right Python dev only 49%).
            # With 3+ parts, one weak part is tolerated — queries often mix a loose role
            # word ("backend developer") with concrete skills ("Python", "system design").
            fails = sum(1 for s in pos if s < AND_FAIL)
            if fails > (1 if len(pos) >= 3 else 0):
                return 0.0
            soft = 0.5 * low + 0.5 * (sum(pos) / len(pos))
            return soft * (0.5 + 0.5 * gate)

        elif op == "OR":
            operands = node.get("operands", [])
            if not operands: return 0.0
            scores = [self.evaluate_node(child, profile) for child in operands]
            return max(scores) if scores else 0.0

        return 0.0

    def score_profile(self, node, profile, whole_query=None):
        """Evaluate and return (score, matched_terms) for one profile.
        If `whole_query` is given (only for queries without NOT/hate), the meaning of the
        full sentence acts as a floor — the parser sometimes drops words ("exploring new
        countries" -> "new countries") and loses the match."""
        self.matched_terms = []
        score = self.evaluate_node(node, profile)
        terms = list(dict.fromkeys(self.matched_terms))  # unique, keep order
        core = _strip_filler(whole_query) if whole_query else ''
        if core:
            whole = 0.9 * self._concept_score(core, 'any', profile, semantic_only=True)
            if whole > score:
                score = whole
        return score, terms


def _has_negation(node):
    """True if the AST contains a NOT or a 'hate' leaf anywhere."""
    if not isinstance(node, dict):
        return False
    if (node.get('operator') or '').upper() == 'NOT' or (node.get('sentiment') or '').lower() == 'hate':
        return True
    kids = list(node.get('operands') or [])
    if node.get('operand'):
        kids.append(node['operand'])
    return any(_has_negation(k) for k in kids)


# ---------------------------------------------------------------------------
# Primary Entry Point: Hybrid Discovery with Mutual Intent + Hard Filters
# ---------------------------------------------------------------------------

def _extract_profile_vec(profile):
    """Safely extract a numpy vector from a profile's interests_embedding."""
    emb = getattr(profile, 'interests_embedding', None)
    if not emb:
        return None
    if isinstance(emb, list) and len(emb) > 0:
        if isinstance(emb[0], list):
            return np.array(emb[0])
        elif isinstance(emb[0], (int, float)):
            return np.array(emb)
    return None


def run_hybrid_discovery(intent_string, profiles_qs, searcher_profile=None):
    """
    Full Matching Pipeline:
      1. HARD FILTERS   – Skip candidates who violate static dealbreakers (zero CPU wasted).
      2. SEARCHER INTENT – Parse searcher's typed intent into AST, score each candidate.
      3. MUTUAL INTENT   – If candidate has a current_intent, parse it into AST and score searcher.
      4. ONBOARDING BLEND – If use_onboarding_data is on, blend baseline persona similarity.
      5. RANK & RETURN   – Sort by final composite score descending.
    
    Returns (sorted_results_list, searcher_ast_dict).
    """
    MATCH_STATS["total_searches"] += 1
    
    # Step 1: Parse the searcher's intent into an AST (one LLM call for the whole batch)
    searcher_ast = parse_intent_to_ast(intent_string)
    print("Parsed Searcher AST:", json.dumps(searcher_ast, indent=2))
    
    evaluator = MatchEvaluator()
    results = []
    # Whole-sentence floor only for single-concept queries: it rescues words the parser
    # dropped ("exploring new countries" -> "new countries"), but for AND/OR/NOT queries
    # it would override the logic ("baking AND deep learning" matching a chef).
    use_whole_query = (searcher_ast.get('operator') or '').upper() == 'LEAF' and not _has_negation(searcher_ast)

    # Pre-cache searcher vector for onboarding blend
    searcher_vec = _extract_profile_vec(searcher_profile) if searcher_profile else None

    # Cache candidate ASTs so we don't re-parse the same intent string
    _candidate_ast_cache = {}

    for candidate in profiles_qs:
        # ---------------------------------------------------------------
        # STAGE 1: HARD FILTER EARLY EXIT
        # ---------------------------------------------------------------
        if searcher_profile:
            passed, reason = passes_hard_filters(searcher_profile, candidate)
            if not passed:
                print(f"  HARD FILTER SKIP: {getattr(candidate, 'user', '?')} – {reason}")
                continue
            # Also check reverse: does the candidate's hard filters reject the searcher?
            passed_rev, reason_rev = passes_hard_filters(candidate, searcher_profile)
            if not passed_rev:
                print(f"  HARD FILTER SKIP (reverse): {getattr(candidate, 'user', '?')} – {reason_rev}")
                continue

        # ---------------------------------------------------------------
        # STAGE 2: SEARCHER INTENT vs CANDIDATE PROFILE
        # ---------------------------------------------------------------
        searcher_to_cand_score, matched_terms = evaluator.score_profile(
            searcher_ast, candidate,
            whole_query=intent_string if use_whole_query else None,
        )

        # ---------------------------------------------------------------
        # STAGE 3: MUTUAL INTENT (Candidate Intent vs Searcher Profile)
        # ---------------------------------------------------------------
        cand_to_searcher_score = 0.0
        mutual_weight = 1.0  # full weight on searcher direction by default

        cand_intent = (getattr(candidate, 'current_intent', '') or '').strip()
        # Only use the candidate's intent if it's something they actually TYPED recently.
        # (Before: button presses like "Persona Match"/"Random Opposite Gender chat" and
        #  weeks-old searches were parsed by the LLM and dragged real matches down.)
        cand_recent = False
        try:
            from django.utils import timezone as _tz
            from datetime import timedelta as _td
            ls = getattr(candidate, 'last_seen', None)
            cand_recent = bool(ls and ls >= _tz.now() - _td(minutes=30))
        except Exception:
            pass
        if (searcher_profile
            and cand_recent
            and len(cand_intent) > 3
            and not cand_intent.lower().startswith(SYSTEM_INTENT_PREFIXES)):

            # Parse from cache or fresh
            if cand_intent not in _candidate_ast_cache:
                _candidate_ast_cache[cand_intent] = parse_intent_to_ast(cand_intent)
            cand_ast = _candidate_ast_cache[cand_intent]

            cand_to_searcher_score, _ = MatchEvaluator().score_profile(cand_ast, searcher_profile)
            # Both directions matter: average them (50/50 mutual)
            mutual_weight = 0.5

        if mutual_weight < 1.0:
            # Mutual mode: average both directions
            intent_score = (searcher_to_cand_score * mutual_weight) + (cand_to_searcher_score * mutual_weight)
        else:
            # One-directional: only searcher's intent matters
            intent_score = searcher_to_cand_score

        # ---------------------------------------------------------------
        # STAGE 4: ONBOARDING PERSONA BLEND (optional)
        # ---------------------------------------------------------------
        final_score = intent_score

        if searcher_vec is not None:
            cand_vec = _extract_profile_vec(candidate)
            if cand_vec is not None:
                # calibrated like everything else (raw cosine is ~0.74 even for strangers)
                onboarding_sim = calibrate(cosine_similarity(searcher_vec, cand_vec))
                
                # Dynamic weighting: if the user typed a long, specific query, their explicit intent 
                # should dominate (preventing the "opposite match" penalty). If the query is generic, 
                # we rely more on baseline chemistry/similarity.
                if len(intent_string) > 30:
                    intent_weight = 0.90
                elif len(intent_string) > 15:
                    intent_weight = 0.75
                else:
                    intent_weight = 0.60
                    
                sim_weight = 1.0 - intent_weight
                final_score = (intent_score * intent_weight) + (max(0.0, onboarding_sim) * sim_weight)
        
        # Ensure the final score never exceeds 100% (1.0)
        final_score = min(1.0, final_score)

        # ---------------------------------------------------------------
        # STAGE 5: THRESHOLD & COLLECT
        # ---------------------------------------------------------------
        MATCH_STATS["total_candidates_scored"] += 1
        # The searcher's own query must match on its own — mutual intent / profile blend
        # can re-rank real matches but can't turn a non-match into a result.
        if final_score >= MIN_MATCH_SCORE and searcher_to_cand_score >= MIN_MATCH_SCORE:
            results.append({
                "profile": candidate,
                "vector_score": final_score,
                "searcher_to_cand": searcher_to_cand_score,
                "cand_to_searcher": cand_to_searcher_score,
                "matched_terms": matched_terms,
            })

    MATCH_STATS["matches_returned"] += len(results)
    # Sort descending by composite score
    results.sort(key=lambda x: x["vector_score"], reverse=True)
    return results, searcher_ast


def profile_similarity(a, b):
    """
    How similar two people's profiles are (calibrated 0..1) + what they share.
    Used by Persona Match ("Match based on my profile") and offline persona searches.
    No LLM call — uses the stored interest/expertise embeddings and literal overlap.
    """
    a_items = [str(x) for x in (a.interests or [])] + [str(x) for x in (a.expertise_areas or [])]
    b_items = [str(x) for x in (b.interests or [])] + [str(x) for x in (b.expertise_areas or [])]
    shared = [x for x in a_items if any(_word_match(x, y) or _word_match(y, x) for y in b_items)]
    shared = list(dict.fromkeys(shared))

    best = 0.0
    for ea in (getattr(a, 'interests_embedding', None), getattr(a, 'expertise_embedding', None)):
        for va in _vectors(ea):
            for eb in (getattr(b, 'interests_embedding', None), getattr(b, 'expertise_embedding', None)):
                for vb in _vectors(eb):
                    best = max(best, calibrate(cosine_similarity(va, vb)))
    if shared:
        # each literally shared interest is strong evidence
        best = max(best, min(1.0, 0.6 + 0.2 * len(shared)))
    return best, shared


def rank_by_profile_similarity(searcher_profile, candidates):
    """Rank candidates by similarity to the searcher's own profile (hard filters applied)."""
    results = []
    for cand in candidates:
        passed, _ = passes_hard_filters(searcher_profile, cand)
        if not passed:
            continue
        passed_rev, _ = passes_hard_filters(cand, searcher_profile)
        if not passed_rev:
            continue
        score, shared = profile_similarity(searcher_profile, cand)
        if score >= MIN_MATCH_SCORE:
            results.append({"profile": cand, "vector_score": score, "matched_terms": shared})
    results.sort(key=lambda x: x["vector_score"], reverse=True)
    return results


def speaks_any(profile, languages):
    """True if the profile lists any of `languages` (case-insensitive).
    Done in Python because SQLite (the production DB) doesn't support JSON `contains`
    lookups — `languages__contains=[...]` crashed every search that used a language filter."""
    if not languages:
        return True
    mine = {str(x).strip().lower() for x in (getattr(profile, 'languages', None) or [])}
    return any(str(l).strip().lower() in mine for l in languages)


def has_profile_data(profile):
    """True if the profile has anything Persona Match can compare."""
    return bool(profile.interests or profile.expertise_areas
                or getattr(profile, 'interests_embedding', None)
                or getattr(profile, 'expertise_embedding', None))
