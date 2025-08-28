from typing import List, Dict
from sentence_transformers import CrossEncoder
from rag.embeddings import LocalEmbedder
from configs.unified_rag_config import EMBEDDING_MODEL_PATH, RRF_WEIGHT, CROSS_ENCODER_WEIGHT
import numpy as np

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
        tile_overlap: int = 50,            # 타일링 오버랩 토큰 수
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
        self.tile_overlap = tile_overlap

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

    def _create_text_tiles(self, text: str, max_tokens: int) -> List[str]:
        """
        긴 텍스트를 슬라이딩 윈도우로 타일링
        Args:
            text: 원본 텍스트
            max_tokens: 각 타일의 최대 토큰 수
        Returns:
            타일링된 텍스트 리스트
        """
        if not text:
            return [""]
        
        # 텍스트를 토큰으로 변환
        tokens = self.tok.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        # 슬라이딩 윈도우로 타일링
        tiles = []
        step = max_tokens - self.tile_overlap
        
        for i in range(0, len(tokens), step):
            tile_tokens = tokens[i:i + max_tokens]
            tile_text = self.tok.decode(tile_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            tiles.append(tile_text)
            
            if i + max_tokens >= len(tokens):
                break
        
        return tiles

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

        # 3) 롱컨텍스트 타일링 적용
        cross_encoder_scores = []
        for c in candidates:
            ctx = c.get("text", "")
            
            # 긴 컨텍스트는 타일링
            if len(self.tok.encode(ctx, add_special_tokens=False)) > ctx_budget:
                tiles = self._create_text_tiles(ctx, ctx_budget)
                
                # 각 타일별 CE 점수 계산
                tile_pairs = [(q_trim, tile) for tile in tiles]
                tile_scores = self.cross_encoder.predict(tile_pairs, batch_size=self.batch_size)
                tile_scores = [float(s) for s in tile_scores]
                
                # 타일 점수 집계 (최댓값 사용)
                final_score = max(tile_scores)
            else:
                # 짧은 컨텍스트는 기존 방식
                ctx_trim = self._truncate_by_tokens(self.tok, ctx, ctx_budget)
                score = self.cross_encoder.predict([(q_trim, ctx_trim)], batch_size=self.batch_size)
                final_score = float(score[0])
            
            cross_encoder_scores.append(final_score)

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
