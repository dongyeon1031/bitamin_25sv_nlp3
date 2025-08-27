from typing import List, Dict
from sentence_transformers import CrossEncoder
from rag.embeddings import LocalEmbedder
from configs.unified_rag_config import EMBEDDING_MODEL_PATH, RRF_WEIGHT, CROSS_ENCODER_WEIGHT

class UnifiedEnsembleReranker:
    """
    통합 레이어드 Ensemble Reranker (법령 + 보안/경제)
    - 1단계: RRF로 BM25 + Vector 융합
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬
    """

    def __init__(
        self,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_weight: float = RRF_WEIGHT,
        cross_encoder_weight: float = CROSS_ENCODER_WEIGHT,
        max_length: int = 512,             # ★ 1024 → 512 로 변경 (XLM-R 한도)
        device: str = None,
        ce_query_budget: int = 128,        # 질문 토큰 예산
        batch_size: int = 16,              # CE 배치
    ):
        self.cross_encoder = CrossEncoder(
            cross_encoder_model,
            device=device,
            max_length=max_length         # ★ predict()에서 자동 truncation
        )
        self.tok = self.cross_encoder.tokenizer
        self.ce_max_length = max_length
        self.ce_query_budget = ce_query_budget
        self.batch_size = batch_size

        self.rrf_weight = rrf_weight
        self.cross_encoder_weight = cross_encoder_weight

    @staticmethod
    def _truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
        """토큰 기준으로 앞부분만 남겨 안전하게 자르기"""
        if max_tokens <= 0 or not text:
            return ""
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores
        mi, ma = min(scores), max(scores)
        if ma == mi:
            return [0.5] * len(scores)
        return [(s - mi) / (ma - mi) for s in scores]

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 8) -> List[Dict]:
        if not candidates:
            return []

        # 1) RRF 점수
        rrf_scores = [c.get("score", 0.0) for c in candidates]

        # 2) CE 입력 준비: 질문/컨텍스트 토큰 예산 분배 후 트렁케이션
        #    - 질문: ce_query_budget (기본 128)
        #    - 컨텍스트: (max_length - 질문토큰 - 여유 3)
        q_trim = self._truncate_by_tokens(self.tok, query, self.ce_query_budget)
        q_ids = self.tok.encode(q_trim, add_special_tokens=False)
        ctx_budget = max(1, self.ce_max_length - min(len(q_ids), self.ce_query_budget) - 3)

        pairs = []
        for c in candidates:
            ctx = c.get("text", "")
            ctx_trim = self._truncate_by_tokens(self.tok, ctx, ctx_budget)
            pairs.append((q_trim, ctx_trim))

        # 3) CE 점수 (512로 안전하게 잘린 상태)
        ce_raw = self.cross_encoder.predict(pairs, batch_size=self.batch_size)
        cross_encoder_scores = [float(s) for s in ce_raw]

        # 4) 정규화 + 가중합
        rrf_norm = self._normalize_scores(rrf_scores)
        ce_norm = self._normalize_scores(cross_encoder_scores)

        reranked = []
        for i, c in enumerate(candidates):
            final_score = self.rrf_weight * rrf_norm[i] + self.cross_encoder_weight * ce_norm[i]
            cc = c.copy()
            cc["ensemble_score"] = final_score
            cc["rrf_score"] = rrf_norm[i]
            cc["cross_encoder_score"] = ce_norm[i]
            reranked.append(cc)

        reranked.sort(key=lambda x: x["ensemble_score"], reverse=True)
        return reranked[:top_k]

    def filter_by_doc_type(self, query: str, candidates: List[Dict], doc_type: str) -> List[Dict]:
        filtered = [c for c in candidates if c.get("meta", {}).get("doc_type") == doc_type]
        if not filtered:
            return []
        return self.rerank(query, filtered)   # ← 쿼리는 그대로 전달 (OK)
