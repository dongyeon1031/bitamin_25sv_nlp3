from typing import List, Dict
from sentence_transformers import CrossEncoder
from rag.embeddings import LocalEmbedder
from configs.rag_config import EMBEDDING_MODEL_PATH

class SimpleEnsembleReranker:
    """
    레이어드 Ensemble Reranker
    - 1단계: RRF로 BM25 + Vector 융합 (기존 유지)
    - 2단계: RRF + CrossEncoder 가중합으로 재정렬 (추가)
    """
    
    def __init__(self, 
                 cross_encoder_model: str = "BAAI/bge-reranker-base",
                 rrf_weight: float = 0.4,
                 cross_encoder_weight: float = 0.6,
                 max_length: int = 1024,
                 device: str = None):
        """
        Args:
            cross_encoder_model: CrossEncoder 모델명
            rrf_weight: RRF 점수 가중치 (BM25+Vector 융합 결과)
            cross_encoder_weight: CrossEncoder 점수 가중치
            max_length: CrossEncoder 최대 길이
            device: 사용할 디바이스
        """
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=device, max_length=max_length)
        self.rrf_weight = rrf_weight
        self.cross_encoder_weight = cross_encoder_weight
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """점수를 0-1 범위로 정규화"""
        if not scores:
            return scores
        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [0.5] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 8) -> List[Dict]:
        """
        레이어드 Ensemble Reranking 수행
        
        Args:
            query: 질문
            candidates: 후보 문서들 (RRF로 이미 융합된 상태)
            top_k: 반환할 상위 문서 수
            
        Returns:
            재순위화된 문서 리스트
        """
        if not candidates:
            return []
        
        # 1. RRF 점수 추출 (1단계에서 이미 BM25+Vector 융합 완료)
        rrf_scores = [c.get("score", 0.0) for c in candidates]
        
        # 2. CrossEncoder 점수 계산 (2단계)
        pairs = [(query, c["text"]) for c in candidates]
        cross_encoder_scores = [float(s) for s in self.cross_encoder.predict(pairs)]
        
        # 3. 점수 정규화
        rrf_normalized = self._normalize_scores(rrf_scores)
        cross_encoder_normalized = self._normalize_scores(cross_encoder_scores)
        
        # 4. RRF + CrossEncoder 가중합으로 최종 점수 계산
        ensemble_scores = []
        for i in range(len(candidates)):
            final_score = (
                self.rrf_weight * rrf_normalized[i] +
                self.cross_encoder_weight * cross_encoder_normalized[i]
            )
            ensemble_scores.append(final_score)
        
        # 5. 재순위화
        reranked = []
        for i, (candidate, ensemble_score) in enumerate(zip(candidates, ensemble_scores)):
            new_candidate = candidate.copy()
            new_candidate["ensemble_score"] = ensemble_score
            new_candidate["rrf_score"] = rrf_normalized[i]
            new_candidate["cross_encoder_score"] = cross_encoder_normalized[i]
            reranked.append(new_candidate)
        
        # Ensemble 점수로 정렬
        reranked.sort(key=lambda x: x["ensemble_score"], reverse=True)
        
        return reranked[:top_k]
