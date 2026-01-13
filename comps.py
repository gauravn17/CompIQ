"""
Refactored Comparables Agent with progress tracking and better modularity.
"""
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
import time
import os
import logging
import numpy as np
from openai import OpenAI
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type aliases
TargetCompany = Dict[str, str]
ComparableCompany = Dict[str, Any]
ProgressCallback = Callable[[str, int], None]


class ComparablesAgent:
    """
    Agent for finding comparable public companies.
    Refactored for better modularity and real-time progress tracking.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        min_required: int = 3,
        max_allowed: int = 10,
        max_attempts: int = 3
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.min_required = min_required
        self.max_allowed = max_allowed
        self.max_attempts = max_attempts
        self.validator = PublicStatusValidator(self.client)
    
    def find_comparables(
        self,
        target: TargetCompany,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Dict[str, Any]:
        """
        Main method to find comparable companies.
        
        Args:
            target: Target company information
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dict with 'comparables' and 'metadata' keys
        """
        def update_progress(step: str, progress: int):
            if progress_callback:
                progress_callback(step, progress)
            logger.info(f"{step} ({progress}%)")
        
        metadata = {
            "target": target["name"],
            "timestamp": datetime.now().isoformat(),
            "rejected_companies": [],
            "validation_method": "dynamic_llm"
        }
        
        # Step 1: Analyze target (10-20%)
        update_progress("Analyzing target company", 10)
        analysis = self._analyze_target(target)
        metadata["analysis"] = analysis
        update_progress("Analysis complete", 20)
        
        # Step 2: Create embeddings (20-30%)
        update_progress("Creating semantic embeddings", 25)
        target_norm = self._normalize_description(target["description"], analysis)
        target_embedding = self._embed_texts([target_norm])[0]
        update_progress("Embeddings created", 30)
        
        # Step 3: Generate and validate candidates (30-90%)
        best_comps, best_rejected = [], []
        
        for attempt in range(1, self.max_attempts + 1):
            use_broader = attempt > 1 and len(best_comps) < self.min_required
            
            progress_start = 30 + (attempt - 1) * 20
            progress_end = progress_start + 20
            
            update_progress(f"Generating candidates (attempt {attempt}/{self.max_attempts})", progress_start)
            
            candidates = self._generate_candidates(
                target, analysis, 25, attempt, use_broader
            )
            
            if not candidates:
                continue
            
            update_progress(f"Validating {len(candidates)} candidates", progress_start + 5)
            
            comps, rejected = self._validate_and_rank(
                candidates,
                analysis,
                target_embedding,
                target["description"]
            )
            
            update_progress(f"Found {len(comps)} valid comparables", progress_end)
            
            if len(comps) >= self.min_required:
                metadata["rejected_companies"] = rejected
                update_progress("Search complete", 100)
                return {
                    "comparables": comps,
                    "metadata": metadata
                }
            
            if len(comps) > len(best_comps):
                best_comps, best_rejected = comps, rejected
        
        # Return best effort
        metadata["rejected_companies"] = best_rejected
        update_progress("Search complete (partial results)", 100)
        return {
            "comparables": best_comps,
            "metadata": metadata
        }
    
    def _analyze_target(self, target: TargetCompany) -> Dict[str, Any]:
        """Analyze target company to extract key characteristics."""
        prompt = f"""
You are an expert investment analyst. Analyze this company deeply.

COMPANY:
Name: {target['name']}
Description: {target['description']}
Primary SIC: {target.get('primary_sic', 'Not provided')}

Return ONLY valid JSON:
{{
  "specialization_level": 0.0-1.0,
  "core_focus_areas": ["term1", "term2", ...],
  "business_model": "consulting|software_vendor|managed_services|hardware|platform|hybrid|other",
  "key_differentiators": ["diff1", "diff2", ...],
  "exclusion_criteria": {{
    "avoid_company_types": ["type1", ...],
    "avoid_characteristics": ["char1", ...]
  }},
  "ideal_comparable_profile": "One sentence describing ideal comparable"
}}
"""
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            analysis = self._safe_parse_json(resp.choices[0].message.content)
            
            # Ensure required fields
            analysis.setdefault("specialization_level", 0.5)
            analysis.setdefault("core_focus_areas", [])
            analysis.setdefault("business_model", "other")
            analysis.setdefault("key_differentiators", [])
            analysis.setdefault("exclusion_criteria", {
                "avoid_company_types": [],
                "avoid_characteristics": []
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing target: {e}")
            return {
                "specialization_level": 0.5,
                "core_focus_areas": [],
                "business_model": "other",
                "key_differentiators": [],
                "exclusion_criteria": {"avoid_company_types": [], "avoid_characteristics": []}
            }
    
    def _generate_candidates(
        self,
        target: TargetCompany,
        analysis: Dict[str, Any],
        max_candidates: int,
        attempt: int,
        use_broader_search: bool
    ) -> List[ComparableCompany]:
        """Generate candidate comparable companies."""
        specialization = analysis["specialization_level"]
        focus_areas = analysis["core_focus_areas"]
        business_model = analysis["business_model"]
        
        if use_broader_search:
            search_strategy = f"""
BROADER SEARCH MODE:
1. Companies in RELATED industries: {', '.join(focus_areas[:5])}
2. Similar business model: {business_model}
3. Include adjacent/upstream/downstream industries
"""
        elif specialization >= 0.7:
            search_strategy = f"""
HIGHLY SPECIALIZED TARGET:
1. >50% revenue from: {', '.join(focus_areas[:5])}
2. Business model: {business_model}
3. AVOID diversified conglomerates
"""
        else:
            search_strategy = f"""
MODERATE TARGET:
1. >30% exposure to: {', '.join(focus_areas[:5])}
2. Business model: {business_model}
"""
        
        prompt = f"""
Find {max_candidates} CURRENTLY PUBLICLY TRADED comparable companies.

TARGET: {target['name']}
Description: {target['description']}

{search_strategy}

CRITICAL: Only suggest companies currently trading. Exclude acquired/delisted companies.

Return JSON array:
[
  {{
    "name": "Company Name",
    "url": "https://...",
    "exchange": "NYSE/NASDAQ/etc",
    "ticker": "TICK",
    "business_activity": "Description",
    "customer_segment": "Who they serve",
    "SIC_industry": "Industry classification",
    "revenue_focus_explanation": "How this matches target"
  }}
]
"""
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4000
            )
            
            candidates = self._safe_parse_json(resp.choices[0].message.content)
            
            if not isinstance(candidates, list):
                return []
            
            # Filter out target company
            target_name_lower = target['name'].lower()
            candidates = [
                c for c in candidates
                if target_name_lower not in c.get("name", "").lower()
            ]
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return []
    
    def _validate_and_rank(
        self,
        candidates: List[ComparableCompany],
        analysis: Dict[str, Any],
        target_embedding: np.ndarray,
        target_description: str
    ) -> Tuple[List[ComparableCompany], List[Dict[str, Any]]]:
        """Validate and rank candidate companies."""
        all_rejected = []
        
        # Basic validation
        valid_candidates = [
            c for c in candidates
            if self._is_valid_company_data(c)
        ]
        
        # Public status validation
        public_valid, public_rejected = self.validator.validate_companies(valid_candidates)
        all_rejected.extend(public_rejected)
        
        # Normalize descriptions and score
        for comp in public_valid:
            if "normalized_description" not in comp:
                text = f"{comp.get('business_activity', '')} {comp.get('customer_segment', '')}"
                comp["normalized_description"] = self._normalize_description(text, analysis)
            
            result = self._score_comparable(
                comp, analysis, target_embedding, target_description
            )
            comp["validation_score"] = result["score"]
            comp["score_breakdown"] = result["breakdown"]
        
        # Sort and filter
        scored = sorted(public_valid, key=lambda x: x["validation_score"], reverse=True)
        
        # Apply score thresholds
        thresholds = [5.0, 4.0, 3.0] if analysis.get("specialization_level", 0.5) >= 0.7 else [4.0, 3.0, 2.0]
        for t in thresholds:
            filtered = [c for c in scored if c["validation_score"] >= t]
            if len(filtered) >= self.min_required:
                return filtered[:self.max_allowed], all_rejected
        
        return scored[:self.max_allowed], all_rejected
    
    def _normalize_description(self, description: str, analysis: Dict[str, Any]) -> str:
        """Normalize description for comparison."""
        focus = ", ".join(analysis.get("core_focus_areas", [])[:5])
        
        prompt = f"""
Rewrite into factual comparable profile focusing on PRIMARY revenue activities.
Context: {focus}
Description: {description}
Return ONE paragraph (3-5 sentences).
"""
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error normalizing: {e}")
            return description
    
    def _score_comparable(
        self,
        comp: ComparableCompany,
        analysis: Dict[str, Any],
        target_embedding: np.ndarray,
        target_description: str
    ) -> Dict[str, Any]:
        """Score how well a comparable matches the target."""
        score = 1.0
        breakdown = {"valid_public_operating": 1.0}
        
        comp_normalized = comp.get("normalized_description", comp.get("business_activity", ""))
        
        # Semantic similarity
        try:
            comp_embedding = self._embed_texts([comp_normalized])[0]
            semantic_sim = self._cosine_similarity(target_embedding, comp_embedding)
            specialization = analysis.get("specialization_level", 0.5)
            weight = 3.0 + (specialization * 2.0)
            score += semantic_sim * weight
            breakdown["semantic_similarity"] = f"{semantic_sim:.3f} (weighted {weight:.1f}x)"
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            breakdown["semantic_similarity"] = "error"
        
        # Focus area overlap
        focus_areas = analysis.get("core_focus_areas", [])
        if focus_areas:
            text_lower = comp_normalized.lower()
            matches = sum(1 for a in focus_areas if a.lower() in text_lower)
            focus_score = matches / len(focus_areas)
            score += focus_score * 1.5
            breakdown["focus_overlap"] = f"{focus_score:.2f}"
        
        # Penalties
        if comp.get("_caveat"):
            score -= 0.5
            breakdown["caveat"] = comp["_caveat"]
        
        if comp.get("_needs_verification"):
            score -= 0.25
            breakdown["needs_verification"] = True
        
        return {"score": round(score, 3), "breakdown": breakdown}
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings."""
        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return np.array([d.embedding for d in resp.data])
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), 1536))
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm_product) if norm_product > 0 else 0.0
    
    @staticmethod
    def _is_valid_company_data(comp: ComparableCompany) -> bool:
        """Validate company data completeness."""
        required = ["name", "url", "exchange", "ticker", "business_activity"]
        for field in required:
            val = comp.get(field, "")
            if not isinstance(val, str) or not val.strip():
                return False
        return True
    
    @staticmethod
    def _safe_parse_json(text: str) -> Any:
        """Parse JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        text_clean = text.strip()
        
        # Remove markdown
        if text_clean.startswith("```"):
            lines = text_clean.split("\n")
            text_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        # Extract array
        start = text_clean.find("[")
        end = text_clean.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_clean[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        # Extract object
        start = text_clean.find("{")
        end = text_clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_clean[start:end + 1])
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Could not parse JSON from response")
        return [] if "[" in text else {}


class PublicStatusValidator:
    """Validates whether companies are currently publicly traded."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def validate_companies(
        self,
        companies: List[ComparableCompany]
    ) -> Tuple[List[ComparableCompany], List[Dict[str, Any]]]:
        """Validate a list of companies."""
        if not companies:
            return [], []
        
        logger.info(f"Verifying public status for {len(companies)} companies...")
        
        verifications = self._verify_batch(companies)
        
        valid = []
        rejected = []
        
        for comp, verification in zip(companies, verifications):
            ticker = comp.get("ticker", "N/A")
            status = verification.get("status", "UNCERTAIN")
            is_public = verification.get("is_publicly_traded")
            
            if is_public == True and status == "ACTIVE":
                material_changes = verification.get("material_changes")
                if material_changes:
                    comp["_caveat"] = f"Material change: {material_changes}"
                valid.append(comp)
                
            elif is_public == False or status in ["ACQUIRED", "MERGED", "DELISTED", "PRIVATE"]:
                rejected.append({
                    "company": comp,
                    "status": status,
                    "reason": verification.get("reason", "No longer publicly traded"),
                    "acquirer": verification.get("acquirer"),
                    "date": verification.get("date_changed")
                })
                
            elif status == "UNCERTAIN":
                rejected.append({
                    "company": comp,
                    "status": "UNCERTAIN",
                    "reason": "Could not confirm status - manual verification required"
                })
            else:
                comp["_needs_verification"] = True
                comp["_verification_note"] = verification.get("reason", "Status uncertain")
                valid.append(comp)
        
        logger.info(f"Results: {len(valid)} valid, {len(rejected)} rejected")
        return valid, rejected
    
    def _verify_batch(
        self,
        companies: List[ComparableCompany],
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Verify companies in batches."""
        results = []
        
        for i in range(0, len(companies), batch_size):
            batch = companies[i:i + batch_size]
            
            company_list = "\n".join([
                f"{j+1}. {c.get('name', 'Unknown')} (Ticker: {c.get('ticker', 'N/A')}, Exchange: {c.get('exchange', 'N/A')})"
                for j, c in enumerate(batch)
            ])
            
            prompt = f"""
Verify CURRENT trading status of each company:
{company_list}

For each: Is it currently trading? Has it been acquired/merged/delisted?

Return JSON array:
[
  {{
    "ticker": "TICK",
    "name": "Company Name",
    "is_publicly_traded": true/false/null,
    "status": "ACTIVE|ACQUIRED|MERGED|DELISTED|PRIVATE|UNCERTAIN",
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "Explanation if not active",
    "acquirer": "Acquirer name if applicable",
    "date_changed": "YYYY-MM-DD if known",
    "material_changes": "Major business changes"
  }}
]
"""
            
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                batch_results = self._safe_parse_json(resp.choices[0].message.content)
                
                if isinstance(batch_results, list) and len(batch_results) == len(batch):
                    results.extend(batch_results)
                else:
                    # Fallback to uncertain
                    results.extend([{
                        "ticker": c.get("ticker", ""),
                        "is_publicly_traded": None,
                        "status": "UNCERTAIN",
                        "confidence": "LOW"
                    } for c in batch])
                    
            except Exception as e:
                logger.error(f"Error in batch verification: {e}")
                results.extend([{
                    "ticker": c.get("ticker", ""),
                    "is_publicly_traded": None,
                    "status": "UNCERTAIN",
                    "confidence": "LOW"
                } for c in batch])
            
            if i + batch_size < len(companies):
                time.sleep(0.5)
        
        return results
    
    @staticmethod
    def _safe_parse_json(text: str) -> Any:
        """Parse JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            text_clean = text.strip()
            if text_clean.startswith("```"):
                lines = text_clean.split("\n")
                text_clean = "\n".join(lines[1:-1])
            
            start = text_clean.find("[")
            end = text_clean.rfind("]")
            if start != -1 and end != -1:
                try:
                    return json.loads(text_clean[start:end + 1])
                except:
                    pass
            
            return []

if __name__ == "__main__":
    main()
