import pytest
import numpy as np
import os
import shutil
from srm import SemanticRoutingMemory, SRMConfig


# -----------------------------------------------------------------------------
# Fixtures (Test Ortamı Hazırlığı)
# -----------------------------------------------------------------------------

@pytest.fixture
def basic_config():
    """Temel, hafif bir konfigürasyon."""
    return SRMConfig(
        d=8,  # Küçük boyut test hızı için
        K=4,  # Az centroid
        m=2,  # Multi-probe
        top_k=5,
        max_candidates=20,
        store_item_embeddings=True,
        embeddings_dtype="float32",
        seed=42
    )


@pytest.fixture
def srm(basic_config):
    """Her test için temiz bir SRM instance'ı döndürür."""
    return SemanticRoutingMemory(basic_config)


@pytest.fixture
def synthetic_data():
    """Testler için rastgele veri üretir."""
    np.random.seed(42)
    # 8 boyutlu, 100 satırlık veri
    X = np.random.rand(100, 8).astype(np.float32)
    # L2 normalize et (SRM genellikle normalize veri sever)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# -----------------------------------------------------------------------------
# 1. Initialization & Validation Tests
# -----------------------------------------------------------------------------

def test_config_validation():
    """Hatalı konfigürasyonların engellendiğini test eder."""

    # 1. Negatif boyut hatası
    # Config objesi oluşturulurken değil, Memory başlatılırken hata vermeli
    with pytest.raises(ValueError):
        cfg = SRMConfig(d=-5)
        SemanticRoutingMemory(cfg)  # <-- Hata burada fırlatılacak

    # 2. K < 2 hatası
    with pytest.raises(ValueError):
        cfg = SRMConfig(d=10, K=1)
        SemanticRoutingMemory(cfg)

    # 3. m > K hatası
    with pytest.raises(ValueError):
        cfg = SRMConfig(d=10, K=5, m=10)
        SemanticRoutingMemory(cfg)

def test_initialization(srm):
    """SRM'in boş haldeyken doğru başladığını test eder."""
    assert srm.codebook is None
    assert len(srm._ids) == 0
    assert srm._emb_cursor == 0


# -----------------------------------------------------------------------------
# 2. Core Workflow Tests (Train -> Add -> Query)
# -----------------------------------------------------------------------------

def test_fit_codebook(srm, synthetic_data):
    """K-Means eğitiminin çalıştığını test eder."""
    srm.fit_codebook(synthetic_data)

    assert srm.codebook is not None
    # Codebook şekli (K, d) olmalı
    assert srm.codebook.shape == (srm.cfg.K, srm.cfg.d)

    # Codebook normalize edilmiş olmalı (pre_normalize=True varsayılan)
    norms = np.linalg.norm(srm.codebook, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_add_items(srm, synthetic_data):
    """Veri eklemenin doğru çalıştığını test eder."""
    srm.fit_codebook(synthetic_data)

    ids = [f"item_{i}" for i in range(len(synthetic_data))]
    srm.add_items(synthetic_data, ids=ids)

    # ID sayısı tutmalı
    assert len(srm._ids) == len(synthetic_data)

    # Embedding cursor ilerlemiş olmalı
    assert srm._emb_cursor == len(synthetic_data)

    # Bucket'lara dağılım kontrolü (her şey tek bucket'a gitmemeli)
    total_items_in_buckets = sum(len(b) for b in srm._buckets)
    assert total_items_in_buckets == len(synthetic_data)


def test_query(srm, synthetic_data):
    """Sorgu mekanizmasının sonuç döndürdüğünü test eder."""
    srm.fit_codebook(synthetic_data)
    srm.add_items(synthetic_data)

    # Veri setinden birini sorgu olarak kullanalım
    query_vec = synthetic_data[0]

    # Kendisini bulması lazım (recall check)
    results = srm.query(query_vec, top_k=5)

    assert "ids" in results
    assert "scores" in results
    assert len(results["ids"]) > 0

    # En yüksek skorlu sonuç kendisi olmalı (veya çok yakın biri)
    # Cosine similarity olduğu için 1.0'a yakın olmalı
    assert results["scores"][0] > 0.99


def test_payloads(basic_config, synthetic_data):
    """Metadata (payload) saklama test edilir."""
    basic_config.store_payloads = True
    srm = SemanticRoutingMemory(basic_config)
    srm.fit_codebook(synthetic_data)

    payloads = [{"info": i} for i in range(len(synthetic_data))]
    srm.add_items(synthetic_data, payloads=payloads)

    results = srm.query(synthetic_data[0], return_payloads=True)
    assert results["payloads"] is not None
    assert results["payloads"][0]["info"] == 0  # İlk eleman 0. index


# -----------------------------------------------------------------------------
# 3. Persistence Tests (Save & Load)
# -----------------------------------------------------------------------------

def test_save_and_load(srm, synthetic_data, tmp_path):
    """Sistemin diske kaydedilip geri yüklenebildiğini test eder."""
    # 1. Setup & Train
    srm.fit_codebook(synthetic_data)
    custom_ids = [f"id_{i}" for i in range(100)]
    srm.add_items(synthetic_data, ids=custom_ids)

    # 2. Save
    save_dir = tmp_path / "srm_test_save"
    srm.save(str(save_dir))

    # Dosyalar oluştu mu?
    assert (save_dir / "config.json").exists()
    assert (save_dir / "routing.pkl").exists()
    assert (save_dir / "embeddings.npy").exists()

    # 3. Load
    loaded_srm = SemanticRoutingMemory.load(str(save_dir))

    # 4. Verify
    assert loaded_srm.cfg.d == srm.cfg.d
    assert len(loaded_srm._ids) == len(srm._ids)
    assert loaded_srm._ids[0] == "id_0"

    # Codebook eşit mi?
    np.testing.assert_array_almost_equal(loaded_srm.codebook, srm.codebook)


# -----------------------------------------------------------------------------
# 4. Edge Cases & Errors
# -----------------------------------------------------------------------------

def test_add_without_fit(srm, synthetic_data):
    """Codebook eğitilmeden veri eklenirse hata vermeli."""
    with pytest.raises(RuntimeError):
        srm.add_items(synthetic_data)


def test_dimension_mismatch(srm, synthetic_data):
    """Yanlış boyutta veri eklenirse hata vermeli."""
    srm.fit_codebook(synthetic_data)

    wrong_dim_data = np.random.rand(10, 999)  # d=8 bekleniyor
    with pytest.raises(ValueError):
        srm.add_items(wrong_dim_data)


def test_empty_query_result(srm):
    """Boş sisteme sorgu atıldığında boş sonuç dönmeli."""
    # Codebook bile fit edilmemişse
    res = srm.query(np.random.rand(8))
    assert len(res["ids"]) == 0
    assert res["n_candidates"] == 0