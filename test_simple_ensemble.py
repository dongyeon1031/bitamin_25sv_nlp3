#!/usr/bin/env python3
"""
레이어드 Ensemble Reranker 성능 테스트
"""

from rag.retriever import HybridRetriever
from rag.simple_ensemble_reranker import SimpleEnsembleReranker
from sentence_transformers import CrossEncoder
from configs.rag_config import (
    RRF_WEIGHT, CROSS_ENCODER_WEIGHT, CROSS_ENCODER_MODEL, FINAL_CONTEXT_K
)

def test_layered_ensemble():
    """레이어드 Ensemble Reranker vs 단일 CrossEncoder 비교"""
    
    print("🔍 레이어드 Ensemble Reranker vs 단일 CrossEncoder 비교")
    print("=" * 70)
    print("1단계: RRF로 BM25 + Vector 융합")
    print("2단계: RRF + CrossEncoder 가중합으로 재정렬")
    
    # 테스트 질문
    test_query = "전자금융거래법 제2조에서 정의하는 전자금융거래란 무엇인가?"
    
    # 초기화
    retriever = HybridRetriever()
    layered_ensemble = SimpleEnsembleReranker(
        cross_encoder_model=CROSS_ENCODER_MODEL,
        rrf_weight=RRF_WEIGHT,
        cross_encoder_weight=CROSS_ENCODER_WEIGHT
    )
    single_reranker = CrossEncoder(CROSS_ENCODER_MODEL)
    
    # 검색 결과 (1단계: RRF로 이미 융합된 상태)
    candidates = retriever.retrieve(test_query, merge_top_k=20)
    print(f"검색 결과 (RRF 융합): {len(candidates)}개")
    
    if not candidates:
        print("검색 결과가 없습니다.")
        return
    
    # 레이어드 Ensemble Reranker 결과 (2단계)
    ensemble_results = layered_ensemble.rerank(test_query, candidates, top_k=FINAL_CONTEXT_K)
    
    # 단일 CrossEncoder 결과
    pairs = [(test_query, c["text"]) for c in candidates[:FINAL_CONTEXT_K]]
    cross_encoder_scores = single_reranker.predict(pairs)
    single_results = []
    for i, (candidate, score) in enumerate(zip(candidates[:FINAL_CONTEXT_K], cross_encoder_scores)):
        single_results.append({
            "text": candidate["text"],
            "rerank_score": float(score)
        })
    single_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # 결과 비교
    print(f"\n📊 결과 비교:")
    print(f"가중치: RRF={RRF_WEIGHT}, CrossEncoder={CROSS_ENCODER_WEIGHT}")
    
    print(f"\n레이어드 Ensemble Top 3:")
    for i, doc in enumerate(ensemble_results[:3], 1):
        print(f"  {i}. Score: {doc['ensemble_score']:.3f} | {doc['text'][:80]}...")
        print(f"     RRF: {doc['rrf_score']:.3f}, CrossEncoder: {doc['cross_encoder_score']:.3f}")
        
    print(f"\n단일 CrossEncoder Top 3:")
    for i, doc in enumerate(single_results[:3], 1):
        print(f"  {i}. Score: {doc['rerank_score']:.3f} | {doc['text'][:80]}...")
    
    print(f"\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_layered_ensemble()
